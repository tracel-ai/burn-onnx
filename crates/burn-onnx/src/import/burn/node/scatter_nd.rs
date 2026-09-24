use super::prelude::*;
use crate::burn::TensorKind;
use onnx_ir::scatter_nd::ScatterNDReduction;

impl NodeCodegen for onnx_ir::scatter_nd::ScatterNDNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let data_arg = self.inputs.first().unwrap();
        let indices_arg = &self.inputs[1];
        let updates_arg = &self.inputs[2];
        let output = arg_to_ident(self.outputs.first().unwrap());

        let data = scope.arg(data_arg);
        let indices = scope.arg(indices_arg);
        let updates = scope.arg(updates_arg);

        let (data_tensor, indices_tensor) = match (&data_arg.ty, &indices_arg.ty) {
            (ArgType::Tensor(d), ArgType::Tensor(i)) => (d, i),
            _ => {
                let msg = format!(
                    "ScatterND node '{}': data and indices inputs must be tensors",
                    self.name
                );
                return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
            }
        };
        let data_kind = TensorKind::from(data_tensor.dtype);
        let indices_rank = indices_tensor.rank;

        if matches!(data_kind, TensorKind::Bool)
            && !matches!(self.config.reduction, ScatterNDReduction::None)
        {
            let msg = format!(
                "ScatterND node '{}': {:?} reduction is not supported for bool tensors",
                self.name, self.config.reduction
            );
            return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
        }

        let update_op = match self.config.reduction {
            ScatterNDReduction::None => quote! { burn::tensor::IndexingUpdateOp::Assign },
            ScatterNDReduction::Add => quote! { burn::tensor::IndexingUpdateOp::Add },
            ScatterNDReduction::Mul => quote! { burn::tensor::IndexingUpdateOp::Mul },
            ScatterNDReduction::Max => quote! { burn::tensor::IndexingUpdateOp::Max },
            ScatterNDReduction::Min => quote! { burn::tensor::IndexingUpdateOp::Min },
        };

        let normalize =
            indexing_helpers::negative_index_normalize(&data, &indices, indices_rank, 0);

        if matches!(data_kind, TensorKind::Bool) {
            // Native scatter_nd panics for bool tensors; round-trip through i64.
            quote! {
                let #output = {
                    #normalize
                    #data.int()
                        .cast(burn::tensor::DType::I64)
                        .scatter_nd(
                            indices_norm,
                            #updates.int().cast(burn::tensor::DType::I64),
                            #update_op,
                        )
                        .bool()
                };
            }
        } else {
            quote! {
                let #output = {
                    #normalize
                    #data.scatter_nd(indices_norm, #updates, #update_op)
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::scatter_nd::{ScatterNDConfig, ScatterNDNodeBuilder, ScatterNDReduction};

    #[test]
    fn test_scatter_nd_none() {
        let config = ScatterNDConfig::new(ScatterNDReduction::None);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1>,
            indices: Tensor<2, Int>,
            updates: Tensor<1>,
        ) -> Tensor<1> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                data.scatter_nd(indices_norm, updates, burn::tensor::IndexingUpdateOp::Assign)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_add() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Add);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1>,
            indices: Tensor<2, Int>,
            updates: Tensor<1>,
        ) -> Tensor<1> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                data.scatter_nd(indices_norm, updates, burn::tensor::IndexingUpdateOp::Add)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_mul() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Mul);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1>,
            indices: Tensor<2, Int>,
            updates: Tensor<1>,
        ) -> Tensor<1> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                data.scatter_nd(indices_norm, updates, burn::tensor::IndexingUpdateOp::Mul)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_max() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Max);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1>,
            indices: Tensor<2, Int>,
            updates: Tensor<1>,
        ) -> Tensor<1> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                data.scatter_nd(indices_norm, updates, burn::tensor::IndexingUpdateOp::Max)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_min() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Min);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1>,
            indices: Tensor<2, Int>,
            updates: Tensor<1>,
        ) -> Tensor<1> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                data.scatter_nd(indices_norm, updates, burn::tensor::IndexingUpdateOp::Min)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_int() {
        let config = ScatterNDConfig::new(ScatterNDReduction::None);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::I64)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2, Int>,
            indices: Tensor<2, Int>,
            updates: Tensor<1, Int>,
        ) -> Tensor<2, Int> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                data.scatter_nd(indices_norm, updates, burn::tensor::IndexingUpdateOp::Assign)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_bool_none() {
        let config = ScatterNDConfig::new(ScatterNDReduction::None);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<2, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                data.int()
                    .cast(burn::tensor::DType::I64)
                    .scatter_nd(
                        indices_norm,
                        updates.int().cast(burn::tensor::DType::I64),
                        burn::tensor::IndexingUpdateOp::Assign,
                    )
                    .bool()
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_bool_add_emits_compile_error() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Add);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<2, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                compile_error!(
                    "ScatterND node 'scatter1': Add reduction is not supported for bool tensors"
                );
                unreachable!()
            };
            output
        }
        "#);
    }
}
