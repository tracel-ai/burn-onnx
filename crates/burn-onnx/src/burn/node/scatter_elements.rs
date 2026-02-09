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
        let dim = self.config.axis.to_tokens();
        let data = scope.arg(self.inputs.first().unwrap());
        let indices = scope.arg(&self.inputs[1]);
        let updates = scope.arg(&self.inputs[2]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        let data_arg = self.inputs.first().unwrap();
        let data_kind = match &data_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input for data"),
        };

        if matches!(data_kind, TensorKind::Bool)
            && !matches!(self.config.reduction, ScatterElementsReduction::None)
        {
            panic!(
                "ScatterElements with {:?} reduction not supported for bool tensors",
                self.config.reduction
            );
        }

        match self.config.reduction {
            // Native scatter for Add reduction
            ScatterElementsReduction::Add => quote! {
                let #output = #data.scatter(#dim, #indices, #updates, burn::tensor::IndexingUpdateOp::Add);
            },

            // For None with numeric types: gather current values, scatter-add the diff
            // At targets: data[p] + (updates[p] - data[p]) = updates[p]
            // Elsewhere: unchanged
            ScatterElementsReduction::None if !matches!(data_kind, TensorKind::Bool) => quote! {
                let #output = {
                    let gathered = #data.clone().gather(#dim, #indices.clone());
                    #data.scatter(#dim, #indices, #updates - gathered, burn::tensor::IndexingUpdateOp::Add)
                };
            },

            // Bool None and Mul/Max/Min need element-by-element loop
            _ => {
                let reduction_body = match self.config.reduction {
                    ScatterElementsReduction::None => quote! {
                        output_flat = output_flat.slice_assign(
                            [target_offset..target_offset + 1],
                            update_val,
                        );
                    },
                    ScatterElementsReduction::Mul => quote! {
                        let existing = output_flat.clone().narrow(0, target_offset, 1);
                        output_flat = output_flat.slice_assign(
                            [target_offset..target_offset + 1],
                            existing.mul(update_val),
                        );
                    },
                    ScatterElementsReduction::Max => quote! {
                        let existing = output_flat.clone().narrow(0, target_offset, 1);
                        let mask = update_val.clone().greater_equal(existing.clone());
                        let result = existing.mask_where(mask, update_val);
                        output_flat = output_flat.slice_assign(
                            [target_offset..target_offset + 1],
                            result,
                        );
                    },
                    ScatterElementsReduction::Min => quote! {
                        let existing = output_flat.clone().narrow(0, target_offset, 1);
                        let mask = update_val.clone().lower_equal(existing.clone());
                        let result = existing.mask_where(mask, update_val);
                        output_flat = output_flat.slice_assign(
                            [target_offset..target_offset + 1],
                            result,
                        );
                    },
                    ScatterElementsReduction::Add => unreachable!(),
                };

                quote! {
                    let #output = {
                        let data_dims = #data.dims();
                        let updates_dims = #updates.dims();
                        let indices_data = #indices.to_data().convert::<i64>();
                        let indices_values: alloc::vec::Vec<i64> = indices_data.into_vec::<i64>().unwrap();
                        let r = data_dims.len();
                        let dim: usize = #dim;

                        let mut data_strides = alloc::vec![1usize; r];
                        for i in (0..r.saturating_sub(1)).rev() {
                            data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                        }
                        let mut idx_strides = alloc::vec![1usize; r];
                        for i in (0..r.saturating_sub(1)).rev() {
                            idx_strides[i] = idx_strides[i + 1] * updates_dims[i + 1];
                        }

                        let total_data: usize = data_dims.iter().product();
                        let total_updates: usize = updates_dims.iter().product();
                        let mut output_flat = #data.reshape([total_data]);
                        let updates_flat = #updates.reshape([total_updates]);

                        for flat_idx in 0..total_updates {
                            let mut remaining = flat_idx;
                            let mut target_offset = 0usize;
                            for d in 0..r {
                                let coord = remaining / idx_strides[d];
                                remaining %= idx_strides[d];
                                if d == dim {
                                    let dim_size = data_dims[d] as i64;
                                    let mut idx = indices_values[flat_idx];
                                    if idx < 0 { idx += dim_size; }
                                    assert!(idx >= 0 && idx < dim_size, "ScatterElements: index out of bounds");
                                    target_offset += idx as usize * data_strides[d];
                                } else {
                                    target_offset += coord * data_strides[d];
                                }
                            }
                            let update_val = updates_flat.clone().narrow(0, flat_idx, 1);
                            #reduction_body
                        }
                        output_flat.reshape(data_dims)
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
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
            data: Tensor<B, 2>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = {
                let gathered = data.clone().gather(0, indices.clone());
                data.scatter(0, indices, updates - gathered, burn::tensor::IndexingUpdateOp::Add)
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
            data: Tensor<B, 2>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = data.scatter(1, indices, updates, burn::tensor::IndexingUpdateOp::Add);
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
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            data: Tensor<B, 2>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = {
                let data_dims = data.dims();
                let updates_dims = updates.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let dim: usize = 0;
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let mut idx_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    idx_strides[i] = idx_strides[i + 1] * updates_dims[i + 1];
                }
                let total_data: usize = data_dims.iter().product();
                let total_updates: usize = updates_dims.iter().product();
                let mut output_flat = data.reshape([total_data]);
                let updates_flat = updates.reshape([total_updates]);
                for flat_idx in 0..total_updates {
                    let mut remaining = flat_idx;
                    let mut target_offset = 0usize;
                    for d in 0..r {
                        let coord = remaining / idx_strides[d];
                        remaining %= idx_strides[d];
                        if d == dim {
                            let dim_size = data_dims[d] as i64;
                            let mut idx = indices_values[flat_idx];
                            if idx < 0 {
                                idx += dim_size;
                            }
                            assert!(
                                idx >= 0 && idx < dim_size,
                                "ScatterElements: index out of bounds"
                            );
                            target_offset += idx as usize * data_strides[d];
                        } else {
                            target_offset += coord * data_strides[d];
                        }
                    }
                    let update_val = updates_flat.clone().narrow(0, flat_idx, 1);
                    let existing = output_flat.clone().narrow(0, target_offset, 1);
                    output_flat = output_flat
                        .slice_assign(
                            [target_offset..target_offset + 1],
                            existing.mul(update_val),
                        );
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        "#);
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
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            data: Tensor<B, 2>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = {
                let data_dims = data.dims();
                let updates_dims = updates.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let dim: usize = 0;
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let mut idx_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    idx_strides[i] = idx_strides[i + 1] * updates_dims[i + 1];
                }
                let total_data: usize = data_dims.iter().product();
                let total_updates: usize = updates_dims.iter().product();
                let mut output_flat = data.reshape([total_data]);
                let updates_flat = updates.reshape([total_updates]);
                for flat_idx in 0..total_updates {
                    let mut remaining = flat_idx;
                    let mut target_offset = 0usize;
                    for d in 0..r {
                        let coord = remaining / idx_strides[d];
                        remaining %= idx_strides[d];
                        if d == dim {
                            let dim_size = data_dims[d] as i64;
                            let mut idx = indices_values[flat_idx];
                            if idx < 0 {
                                idx += dim_size;
                            }
                            assert!(
                                idx >= 0 && idx < dim_size,
                                "ScatterElements: index out of bounds"
                            );
                            target_offset += idx as usize * data_strides[d];
                        } else {
                            target_offset += coord * data_strides[d];
                        }
                    }
                    let update_val = updates_flat.clone().narrow(0, flat_idx, 1);
                    let existing = output_flat.clone().narrow(0, target_offset, 1);
                    let mask = update_val.clone().greater_equal(existing.clone());
                    let result = existing.mask_where(mask, update_val);
                    output_flat = output_flat
                        .slice_assign([target_offset..target_offset + 1], result);
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        "#);
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
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            data: Tensor<B, 2>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = {
                let data_dims = data.dims();
                let updates_dims = updates.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let dim: usize = 0;
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let mut idx_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    idx_strides[i] = idx_strides[i + 1] * updates_dims[i + 1];
                }
                let total_data: usize = data_dims.iter().product();
                let total_updates: usize = updates_dims.iter().product();
                let mut output_flat = data.reshape([total_data]);
                let updates_flat = updates.reshape([total_updates]);
                for flat_idx in 0..total_updates {
                    let mut remaining = flat_idx;
                    let mut target_offset = 0usize;
                    for d in 0..r {
                        let coord = remaining / idx_strides[d];
                        remaining %= idx_strides[d];
                        if d == dim {
                            let dim_size = data_dims[d] as i64;
                            let mut idx = indices_values[flat_idx];
                            if idx < 0 {
                                idx += dim_size;
                            }
                            assert!(
                                idx >= 0 && idx < dim_size,
                                "ScatterElements: index out of bounds"
                            );
                            target_offset += idx as usize * data_strides[d];
                        } else {
                            target_offset += coord * data_strides[d];
                        }
                    }
                    let update_val = updates_flat.clone().narrow(0, flat_idx, 1);
                    let existing = output_flat.clone().narrow(0, target_offset, 1);
                    let mask = update_val.clone().lower_equal(existing.clone());
                    let result = existing.mask_where(mask, update_val);
                    output_flat = output_flat
                        .slice_assign([target_offset..target_offset + 1], result);
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        "#);
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
            data: Tensor<B, 2, Int>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 2, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = {
                let gathered = data.clone().gather(0, indices.clone());
                data.scatter(0, indices, updates - gathered, burn::tensor::IndexingUpdateOp::Add)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool)
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool)
            .output_tensor("output", 1, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            data: Tensor<B, 1, Bool>,
            indices: Tensor<B, 1, Int>,
            updates: Tensor<B, 1, Bool>,
        ) -> Tensor<B, 1, Bool> {
            let output = {
                let data_dims = data.dims();
                let updates_dims = updates.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let dim: usize = 0;
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let mut idx_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    idx_strides[i] = idx_strides[i + 1] * updates_dims[i + 1];
                }
                let total_data: usize = data_dims.iter().product();
                let total_updates: usize = updates_dims.iter().product();
                let mut output_flat = data.reshape([total_data]);
                let updates_flat = updates.reshape([total_updates]);
                for flat_idx in 0..total_updates {
                    let mut remaining = flat_idx;
                    let mut target_offset = 0usize;
                    for d in 0..r {
                        let coord = remaining / idx_strides[d];
                        remaining %= idx_strides[d];
                        if d == dim {
                            let dim_size = data_dims[d] as i64;
                            let mut idx = indices_values[flat_idx];
                            if idx < 0 {
                                idx += dim_size;
                            }
                            assert!(
                                idx >= 0 && idx < dim_size,
                                "ScatterElements: index out of bounds"
                            );
                            target_offset += idx as usize * data_strides[d];
                        } else {
                            target_offset += coord * data_strides[d];
                        }
                    }
                    let update_val = updates_flat.clone().narrow(0, flat_idx, 1);
                    output_flat = output_flat
                        .slice_assign([target_offset..target_offset + 1], update_val);
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        "#);
    }

    #[test]
    #[should_panic(expected = "reduction not supported for bool tensors")]
    fn test_scatter_elements_bool_add_panics() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool)
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool)
            .output_tensor("output", 1, DType::Bool)
            .config(config)
            .build();
        codegen_forward_default(&node);
    }
}
