use super::prelude::*;
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

        match &self.inputs[0].ty {
            ArgType::Tensor(_) => {}
            ArgType::Shape(rank) => {
                return forward_shape(self, data, indices, updates, output, *rank);
            }
            _ => {
                let msg = format!(
                    "ScatterElements node '{}': data input must be a tensor",
                    self.name
                );
                return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
            }
        }

        // On bool tensors burn's Add and Max are a logical or and Mul and Min a logical and,
        // as in NumPy (and so in the ONNX reference).
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
        let prologue = quote! {
            let (data, indices, updates) = (#data, #indices, #updates);
            let axis_size = data.dims()[#axis] as i64;
            let indices = indices
                .cast(burn::tensor::DType::I64)
                .remainder_scalar(axis_size);
        };

        // Element-wise `scatter` implements all five update ops on every backend and, since
        // burn 0.23.0-pre.2 (tracel-ai/burn#6381), accepts indices smaller than `data` on the
        // non-axis dimensions, which is all ONNX requires: each index coordinate maps onto the
        // same data coordinate. burn documents duplicate indices as undefined for Assign.
        //
        // An empty index tensor is a legal ONNX no-op, so it returns `data` without launching
        // a kernel.
        quote! {
            let #output = {
                #prologue
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(#axis, indices, updates, #update_op)
                }
            };
        }
    }
}

/// Scatter into a Shape value on the host. Shape `indices` and `updates` are `[i64; K]`
/// arrays; a runtime tensor is read back.
fn forward_shape(
    node: &onnx_ir::scatter_elements::ScatterElementsNode,
    data: TokenStream,
    indices: TokenStream,
    updates: TokenStream,
    output: Ident,
    rank: usize,
) -> TokenStream {
    let host_values = |arg: &Argument, value: TokenStream| match &arg.ty {
        ArgType::Shape(_) => value,
        _ => quote! { #value.to_data().iter::<i64>() },
    };
    let indices_iter = host_values(&node.inputs[1], quote! { indices });
    let updates_iter = host_values(&node.inputs[2], quote! { updates });
    let update = match &node.config.reduction {
        ScatterElementsReduction::None => quote! { data[index] = update; },
        ScatterElementsReduction::Add => quote! { data[index] += update; },
        ScatterElementsReduction::Mul => quote! { data[index] *= update; },
        ScatterElementsReduction::Max => quote! { data[index] = data[index].max(update); },
        ScatterElementsReduction::Min => quote! { data[index] = data[index].min(update); },
    };
    let rank_usize = rank.to_tokens();
    let rank_i64 = proc_macro2::Literal::i64_unsuffixed(rank as i64);

    // ONNX allows indices down to `-rank`; `rem_euclid` folds them like the tensor path's
    // floored remainder.
    quote! {
        let #output: [i64; #rank_usize] = {
            let (mut data, indices, updates) = (#data, #indices, #updates);
            for (index, update) in core::iter::zip(#indices_iter, #updates_iter) {
                let index = index.rem_euclid(#rank_i64) as usize;
                #update
            }
            data
        };
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
        assert_snapshot!(code, @"
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
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Assign)
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
        assert_snapshot!(code, @"
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
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(1, indices, updates, burn::tensor::IndexingUpdateOp::Add)
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
        assert_snapshot!(code, @"
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
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Mul)
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
        assert_snapshot!(code, @"
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
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Max)
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
        assert_snapshot!(code, @"
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
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Min)
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
        assert_snapshot!(code, @"
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
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Assign)
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
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<1, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[0] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Assign)
                }
            };
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
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            data: Tensor<2, Bool>,
            indices: Tensor<2, Int>,
            updates: Tensor<2, Bool>,
        ) -> Tensor<2, Bool> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[1] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(1, indices, updates, burn::tensor::IndexingUpdateOp::Assign)
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_add() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<1, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                let (data, indices, updates) = (data, indices, updates);
                let axis_size = data.dims()[0] as i64;
                let indices = indices.cast(burn::tensor::DType::I64).remainder_scalar(axis_size);
                if indices.dims().contains(&0) {
                    data
                } else {
                    data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Add)
                }
            };
            output
        }
        ");
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

    #[test]
    fn test_scatter_elements_shape_data() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_shape("data", 3)
            .input_shape("indices", 1)
            .input_shape("updates", 1)
            .output_shape("output", 3)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: [i64; 3], indices: [i64; 1], updates: [i64; 1]) -> [i64; 3] {
            let output: [i64; 3] = {
                let (mut data, indices, updates) = (data, indices, updates);
                for (index, update) in core::iter::zip(indices, updates) {
                    let index = index.rem_euclid(3) as usize;
                    data[index] = update;
                }
                data
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_shape_data_runtime_tensor_indices_max() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Max);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_shape("data", 3)
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::I64)
            .output_shape("output", 3)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: [i64; 3],
            indices: Tensor<1, Int>,
            updates: Tensor<1, Int>,
        ) -> [i64; 3] {
            let output: [i64; 3] = {
                let (mut data, indices, updates) = (data, indices, updates);
                for (index, update) in core::iter::zip(
                    indices.to_data().iter::<i64>(),
                    updates.to_data().iter::<i64>(),
                ) {
                    let index = index.rem_euclid(3) as usize;
                    data[index] = data[index].max(update);
                }
                data
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_shape_data_add() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_shape("data", 3)
            .input_shape("indices", 1)
            .input_shape("updates", 1)
            .output_shape("output", 3)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: [i64; 3], indices: [i64; 1], updates: [i64; 1]) -> [i64; 3] {
            let output: [i64; 3] = {
                let (mut data, indices, updates) = (data, indices, updates);
                for (index, update) in core::iter::zip(indices, updates) {
                    let index = index.rem_euclid(3) as usize;
                    data[index] += update;
                }
                data
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_shape_data_mul() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Mul);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_shape("data", 3)
            .input_shape("indices", 1)
            .input_shape("updates", 1)
            .output_shape("output", 3)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: [i64; 3], indices: [i64; 1], updates: [i64; 1]) -> [i64; 3] {
            let output: [i64; 3] = {
                let (mut data, indices, updates) = (data, indices, updates);
                for (index, update) in core::iter::zip(indices, updates) {
                    let index = index.rem_euclid(3) as usize;
                    data[index] *= update;
                }
                data
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_shape_data_min() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Min);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_shape("data", 3)
            .input_shape("indices", 1)
            .input_shape("updates", 1)
            .output_shape("output", 3)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: [i64; 3], indices: [i64; 1], updates: [i64; 1]) -> [i64; 3] {
            let output: [i64; 3] = {
                let (mut data, indices, updates) = (data, indices, updates);
                for (index, update) in core::iter::zip(indices, updates) {
                    let index = index.rem_euclid(3) as usize;
                    data[index] = data[index].min(update);
                }
                data
            };
            output
        }
        ");
    }
}
