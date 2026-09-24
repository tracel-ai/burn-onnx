use super::prelude::*;
use crate::burn::TensorKind;
use onnx_ir::scatter_elements::ScatterElementsReduction;

impl NodeCodegen for onnx_ir::scatter_elements::ScatterElementsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let axis = self.config.axis.to_tokens();
        let data = scope.arg(self.inputs.first().unwrap());
        let indices = scope.arg(&self.inputs[1]);
        let updates = scope.arg(&self.inputs[2]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        let (data_kind, rank) = match &self.inputs[0].ty {
            ArgType::Tensor(t) => (TensorKind::from(t.dtype), t.rank),
            _ => {
                let msg = format!(
                    "ScatterElements node '{}': data input must be a tensor",
                    self.name
                );
                return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
            }
        };
        let rank_lit = rank.to_tokens();

        if matches!(data_kind, TensorKind::Bool)
            && !matches!(self.config.reduction, ScatterElementsReduction::None)
        {
            let msg = format!(
                "ScatterElements node '{}': {:?} reduction is not supported for bool tensors",
                self.name, self.config.reduction
            );
            return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
        }

        let update_op = match &self.config.reduction {
            ScatterElementsReduction::None => quote! { burn::tensor::IndexingUpdateOp::Assign },
            ScatterElementsReduction::Add => quote! { burn::tensor::IndexingUpdateOp::Add },
            ScatterElementsReduction::Mul => quote! { burn::tensor::IndexingUpdateOp::Mul },
            ScatterElementsReduction::Max => quote! { burn::tensor::IndexingUpdateOp::Max },
            ScatterElementsReduction::Min => quote! { burn::tensor::IndexingUpdateOp::Min },
        };

        // ONNX allows indices down to `-dim_size` along the scatter axis, which burn's
        // indexing does not accept. burn's remainder is floored, so taking it modulo the
        // axis size folds negatives in one op. Indices outside `[-dim_size, dim_size - 1]`
        // are an error per the ONNX spec and stay unchecked here; adding a guard would
        // mean reading the indices back to the host on every forward pass.
        //
        // The graph values are read once up front: every later read is of a local, so
        // no temporary below can shadow a graph value that shares its name.
        //
        // `scatter` has no Assign for bool tensors (only Add, as a logical or), and
        // `scatter_nd` none at all, so bool data round-trips through i64.
        let is_bool = matches!(data_kind, TensorKind::Bool);
        let to_int = is_bool.then(|| {
            quote! {
                let (data, updates) = (
                    data.int().cast(burn::tensor::DType::I64),
                    updates.int().cast(burn::tensor::DType::I64),
                );
            }
        });
        let to_bool = is_bool.then(|| quote! { .bool() });
        let prologue = quote! {
            let (data, indices, updates) = (#data, #indices, #updates);
            #to_int
            let axis_size = data.dims()[#axis] as i64;
            let indices = indices
                .cast(burn::tensor::DType::I64)
                .remainder_scalar(axis_size);
        };

        // Element-wise `scatter` implements all five update ops on every backend, but
        // it requires `indices` to match `data` on every non-axis dimension, while ONNX
        // only bounds them by it. When the shapes line up, which is the common case,
        // the native kernel runs directly.
        let native = quote! { data.scatter(#axis, indices, updates, #update_op) };

        // A rank-1 tensor has no non-axis dimension to disagree on.
        if rank == 1 {
            return quote! {
                let #output = {
                    #prologue
                    #native
                }#to_bool;
            };
        }

        // Otherwise ScatterElements assigns
        //   output[p_0, .., p_{axis-1}, indices[p], p_{axis+1}, .., p_{r-1}] = updates[p]
        // for every p in the index shape, so materializing those coordinates as index
        // tuples turns it into a ScatterND, which accepts any index shape. The non-axis
        // columns are the row-major coordinates of p, recovered from a flat arange.
        //
        // Duplicate indices fold sequentially on the CPU backends but race on cubecl.
        // burn documents duplicates as undefined for Assign on both paths and for Mul,
        // Min and Max on the scatter_nd path.
        let coordinates = quote! {
            let mut strides = [1usize; #rank_lit];
            for d in (0..#rank_lit - 1).rev() {
                strides[d] = strides[d + 1] * idx_dims[d + 1];
            }
            let flat = Tensor::<1, Int>::arange(
                0..n as i64,
                (&self.device, burn::tensor::DType::I64),
            );
            let mut columns: alloc::vec::Vec<Tensor<2, Int>> =
                alloc::vec::Vec::with_capacity(#rank_lit);
            for d in 0..#rank_lit {
                columns.push(if d == #axis {
                    indices.clone().reshape([n, 1])
                } else {
                    flat
                        .clone()
                        .div_scalar(strides[d] as i64)
                        .remainder_scalar(idx_dims[d] as i64)
                        .reshape([n, 1])
                });
            }
            let coordinates = Tensor::cat(columns, 1);
        };

        // An empty index tensor is a legal ONNX no-op, but `scatter_nd` rejects empty
        // indices and `reshape([0, ..])` would read the 0 as "keep the source dim".
        quote! {
            let #output = {
                #prologue
                let idx_dims = indices.dims();
                let data_dims = data.dims();
                let n: usize = idx_dims.iter().product();
                if n == 0 {
                    data
                } else if (0..#rank_lit).all(|d| d == #axis || idx_dims[d] == data_dims[d]) {
                    #native
                } else {
                    #coordinates
                    data.scatter_nd(coordinates, updates.reshape([n]), #update_op)
                }
            }#to_bool;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::scatter_elements::{
        ScatterElementsConfig, ScatterElementsNodeBuilder, ScatterElementsReduction,
    };

    #[test]
    fn test_scatter_elements_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[0] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                let idx_dims = indices.dims();
                let data_dims = data.dims();
                let n: usize = idx_dims.iter().product();
                if n == 0 {
                    data
                } else if (0..2).all(|d| d == 0 || idx_dims[d] == data_dims[d]) {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Assign)
                } else {
                    let mut strides = [1usize; 2];
                    for d in (0..2 - 1).rev() {
                        strides[d] = strides[d + 1] * idx_dims[d + 1];
                    }
                    let flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for d in 0..2 {
                        columns
                            .push(
                                if d == 0 {
                                    indices.clone().reshape([n, 1])
                                } else {
                                    flat.clone()
                                        .div_scalar(strides[d] as i64)
                                        .remainder_scalar(idx_dims[d] as i64)
                                        .reshape([n, 1])
                                },
                            );
                    }
                    let coordinates = Tensor::cat(columns, 1);
                    data.scatter_nd(
                        coordinates,
                        updates.reshape([n]),
                        burn::tensor::IndexingUpdateOp::Assign,
                    )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_add() {
        let config = ScatterElementsConfig::new(1, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[1] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                let idx_dims = indices.dims();
                let data_dims = data.dims();
                let n: usize = idx_dims.iter().product();
                if n == 0 {
                    data
                } else if (0..2).all(|d| d == 1 || idx_dims[d] == data_dims[d]) {
                    data.scatter(1, indices, updates, burn::tensor::IndexingUpdateOp::Add)
                } else {
                    let mut strides = [1usize; 2];
                    for d in (0..2 - 1).rev() {
                        strides[d] = strides[d + 1] * idx_dims[d + 1];
                    }
                    let flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for d in 0..2 {
                        columns
                            .push(
                                if d == 1 {
                                    indices.clone().reshape([n, 1])
                                } else {
                                    flat.clone()
                                        .div_scalar(strides[d] as i64)
                                        .remainder_scalar(idx_dims[d] as i64)
                                        .reshape([n, 1])
                                },
                            );
                    }
                    let coordinates = Tensor::cat(columns, 1);
                    data.scatter_nd(
                        coordinates,
                        updates.reshape([n]),
                        burn::tensor::IndexingUpdateOp::Add,
                    )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_mul() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Mul);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[0] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                let idx_dims = indices.dims();
                let data_dims = data.dims();
                let n: usize = idx_dims.iter().product();
                if n == 0 {
                    data
                } else if (0..2).all(|d| d == 0 || idx_dims[d] == data_dims[d]) {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Mul)
                } else {
                    let mut strides = [1usize; 2];
                    for d in (0..2 - 1).rev() {
                        strides[d] = strides[d + 1] * idx_dims[d + 1];
                    }
                    let flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for d in 0..2 {
                        columns
                            .push(
                                if d == 0 {
                                    indices.clone().reshape([n, 1])
                                } else {
                                    flat.clone()
                                        .div_scalar(strides[d] as i64)
                                        .remainder_scalar(idx_dims[d] as i64)
                                        .reshape([n, 1])
                                },
                            );
                    }
                    let coordinates = Tensor::cat(columns, 1);
                    data.scatter_nd(
                        coordinates,
                        updates.reshape([n]),
                        burn::tensor::IndexingUpdateOp::Mul,
                    )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_max() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Max);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[0] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                let idx_dims = indices.dims();
                let data_dims = data.dims();
                let n: usize = idx_dims.iter().product();
                if n == 0 {
                    data
                } else if (0..2).all(|d| d == 0 || idx_dims[d] == data_dims[d]) {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Max)
                } else {
                    let mut strides = [1usize; 2];
                    for d in (0..2 - 1).rev() {
                        strides[d] = strides[d + 1] * idx_dims[d + 1];
                    }
                    let flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for d in 0..2 {
                        columns
                            .push(
                                if d == 0 {
                                    indices.clone().reshape([n, 1])
                                } else {
                                    flat.clone()
                                        .div_scalar(strides[d] as i64)
                                        .remainder_scalar(idx_dims[d] as i64)
                                        .reshape([n, 1])
                                },
                            );
                    }
                    let coordinates = Tensor::cat(columns, 1);
                    data.scatter_nd(
                        coordinates,
                        updates.reshape([n]),
                        burn::tensor::IndexingUpdateOp::Max,
                    )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_min() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Min);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[0] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                let idx_dims = indices.dims();
                let data_dims = data.dims();
                let n: usize = idx_dims.iter().product();
                if n == 0 {
                    data
                } else if (0..2).all(|d| d == 0 || idx_dims[d] == data_dims[d]) {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Min)
                } else {
                    let mut strides = [1usize; 2];
                    for d in (0..2 - 1).rev() {
                        strides[d] = strides[d + 1] * idx_dims[d + 1];
                    }
                    let flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for d in 0..2 {
                        columns
                            .push(
                                if d == 0 {
                                    indices.clone().reshape([n, 1])
                                } else {
                                    flat.clone()
                                        .div_scalar(strides[d] as i64)
                                        .remainder_scalar(idx_dims[d] as i64)
                                        .reshape([n, 1])
                                },
                            );
                    }
                    let coordinates = Tensor::cat(columns, 1);
                    data.scatter_nd(
                        coordinates,
                        updates.reshape([n]),
                        burn::tensor::IndexingUpdateOp::Min,
                    )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_int() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::I64)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2, Int>,
            indices: Tensor<2, Int>,
            updates: Tensor<2, Int>,
        ) -> Tensor<2, Int> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[0] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                let idx_dims = indices.dims();
                let data_dims = data.dims();
                let n: usize = idx_dims.iter().product();
                if n == 0 {
                    data
                } else if (0..2).all(|d| d == 0 || idx_dims[d] == data_dims[d]) {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Assign)
                } else {
                    let mut strides = [1usize; 2];
                    for d in (0..2 - 1).rev() {
                        strides[d] = strides[d + 1] * idx_dims[d + 1];
                    }
                    let flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for d in 0..2 {
                        columns
                            .push(
                                if d == 0 {
                                    indices.clone().reshape([n, 1])
                                } else {
                                    flat.clone()
                                        .div_scalar(strides[d] as i64)
                                        .remainder_scalar(idx_dims[d] as i64)
                                        .reshape([n, 1])
                                },
                            );
                    }
                    let coordinates = Tensor::cat(columns, 1);
                    data.scatter_nd(
                        coordinates,
                        updates.reshape([n]),
                        burn::tensor::IndexingUpdateOp::Assign,
                    )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<1, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let (data, updates) = (
                    data.int().cast(burn::tensor::DType::I64),
                    updates.int().cast(burn::tensor::DType::I64),
                );
                let axis_size = data.dims()[0] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Assign)
            }
                .bool();
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_rank2() {
        let config = ScatterElementsConfig::new(1, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::Bool(BoolStore::Native))
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2, Bool>,
            indices: Tensor<2, Int>,
            updates: Tensor<2, Bool>,
        ) -> Tensor<2, Bool> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let (data, updates) = (
                    data.int().cast(burn::tensor::DType::I64),
                    updates.int().cast(burn::tensor::DType::I64),
                );
                let axis_size = data.dims()[1] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                let idx_dims = indices.dims();
                let data_dims = data.dims();
                let n: usize = idx_dims.iter().product();
                if n == 0 {
                    data
                } else if (0..2).all(|d| d == 1 || idx_dims[d] == data_dims[d]) {
                    data.scatter(1, indices, updates, burn::tensor::IndexingUpdateOp::Assign)
                } else {
                    let mut strides = [1usize; 2];
                    for d in (0..2 - 1).rev() {
                        strides[d] = strides[d + 1] * idx_dims[d + 1];
                    }
                    let flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for d in 0..2 {
                        columns
                            .push(
                                if d == 1 {
                                    indices.clone().reshape([n, 1])
                                } else {
                                    flat.clone()
                                        .div_scalar(strides[d] as i64)
                                        .remainder_scalar(idx_dims[d] as i64)
                                        .reshape([n, 1])
                                },
                            );
                    }
                    let coordinates = Tensor::cat(columns, 1);
                    data.scatter_nd(
                        coordinates,
                        updates.reshape([n]),
                        burn::tensor::IndexingUpdateOp::Assign,
                    )
                }
            }
                .bool();
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_add_emits_compile_error() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<1, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                compile_error!(
                    "ScatterElements node 'scatter1': Add reduction is not supported for bool tensors"
                );
                unreachable!()
            };
            output
        }
        "#);
    }

    #[test]
    fn test_scatter_elements_inputs_named_like_temporaries() {
        // Every graph value is read before the block binds any temporary, so inputs
        // may share the temporaries' names.
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data_dims", 2, DType::F32)
            .input_tensor("n", 2, DType::I64)
            .input_tensor("coordinates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        assert!(shadow_check_result(&node).is_ok());
    }
}
