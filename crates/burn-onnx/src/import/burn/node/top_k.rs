use super::prelude::*;

impl NodeCodegen for onnx_ir::topk::TopKNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        // TopK has 2 outputs: values and indices
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // TopK has 2 outputs
        let values_output = arg_to_ident(&self.outputs[0]);
        let indices_output = arg_to_ident(&self.outputs[1]);

        // ONNX spec: TopK indices are always int64. Burn's `topk_with_indices`
        // returns the backend's default int element type, which varies across
        // backends, so cast the indices to the ONNX-specified dtype to keep the
        // output dtype faithful to the model regardless of backend default.
        let indices_dtype_tokens = match &self.outputs[1].ty {
            ArgType::Tensor(t) => t.dtype.to_tokens(),
            other => panic!("TopK indices output must be Tensor, got {other:?}"),
        };

        let axis = self.config.axis.to_tokens();

        // Runtime `k` can reach this path in three shapes: a native scalar
        // (if an earlier pass has already scalarized it), a rank-0 scalar
        // tensor, or ONNX opset-10+'s rank-1 single-element tensor. Lower
        // whichever form to a `usize` local in a prelude block so (1) the
        // main topk_with_indices call stays readable and (2) multiple
        // TopK nodes in the same forward() can't collide on `k`.
        let (prelude, k) = match &self.config.k {
            onnx_ir::topk::TopKInput::Static(k_value) => (TokenStream::new(), k_value.to_tokens()),
            onnx_ir::topk::TopKInput::Runtime(r) => {
                let arg = &self.inputs[r.input_index];
                let prelude = match &arg.ty {
                    ArgType::ScalarNative(_) => {
                        let ident = arg_to_ident(arg);
                        // Clamp with `max(0)` before `as usize`. A negative
                        // runtime k is out-of-spec for ONNX TopK; letting a
                        // negative i64 wrap to a huge usize causes OOM or a
                        // cryptic deep-burn panic. Clamp to zero so a
                        // broken model produces an empty top-k result
                        // instead. Adding observability is tracked in
                        // tracel-ai/burn-onnx#328.
                        quote! { let k: usize = (#ident as i64).max(0) as usize; }
                    }
                    ArgType::ScalarTensor(_) | ArgType::Tensor(_) => {
                        let tensor = scope.arg(arg);
                        quote! {
                            let k: usize = {
                                let data = #tensor.to_data().convert::<i64>();
                                data.as_slice::<i64>().unwrap()[0].max(0) as usize
                            };
                        }
                    }
                    other => panic!("TopK k must be a scalar or rank-1 tensor, got {other:?}"),
                };
                (prelude, quote! { k })
            }
        };

        let input = scope.arg(self.inputs.first().unwrap());

        // burn's topk only selects the largest; the smallest k are the head of an
        // ascending sort.
        let select = if self.config.largest {
            quote! { #input.topk_with_indices(#k, #axis) }
        } else {
            quote! {{
                let (values, indices) = #input.sort_with_indices(#axis);
                (values.narrow(#axis, 0, #k), indices.narrow(#axis, 0, #k))
            }}
        };

        quote! {
            let (#values_output, #indices_output) = {
                #prelude
                let (values, indices) = #select;
                (values, indices.cast(#indices_dtype_tokens))
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::topk::{TopKConfig, TopKInput, TopKNodeBuilder};

    #[test]
    fn test_top_k() {
        let config = TopKConfig::new(1, TopKInput::Static(5), true, true);
        let node = TopKNodeBuilder::new("topk1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("values", 2, DType::F32)
            .output_tensor("indices", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> (Tensor<2>, Tensor<2, Int>) {
            let (values, indices) = {
                let (values, indices) = input.topk_with_indices(5, 1);
                (values, indices.cast(burn::tensor::DType::I64))
            };
            (values, indices)
        }
        ");
    }

    #[test]
    fn test_top_k_smallest() {
        let config = TopKConfig::new(1, TopKInput::Static(5), false, true);
        let node = TopKNodeBuilder::new("topk1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("values", 2, DType::F32)
            .output_tensor("indices", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> (Tensor<2>, Tensor<2, Int>) {
            let (values, indices) = {
                let (values, indices) = {
                    let (values, indices) = input.sort_with_indices(1);
                    (values.narrow(1, 0, 5), indices.narrow(1, 0, 5))
                };
                (values, indices.cast(burn::tensor::DType::I64))
            };
            (values, indices)
        }
        ");
    }

    #[test]
    fn data_input_named_like_the_k_local_is_rejected() {
        // `let k: usize = ...` precedes the read of the data input, so a data
        // input named `k` would be read as the usize local.
        let config = TopKConfig::new(
            1,
            TopKInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("count".to_string(), 1)),
            true,
            true,
        );
        let node = TopKNodeBuilder::new("topk_rt")
            .input_tensor("k", 2, DType::F32)
            .input_tensor("count", 1, DType::I64)
            .output_tensor("values", 2, DType::F32)
            .output_tensor("indices", 2, DType::I64)
            .config(config)
            .build();
        let error = shadow_check_result(&node).unwrap_err();
        assert_eq!(error.name(), Some("k"));
    }

    #[test]
    fn test_top_k_runtime_k() {
        // Opset 10+ passes k as a runtime 1D single-element tensor.
        let config = TopKConfig::new(
            1,
            TopKInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("k".to_string(), 1)),
            true,
            true,
        );
        let node = TopKNodeBuilder::new("topk_rt")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("k", 1, DType::I64)
            .output_tensor("values", 2, DType::F32)
            .output_tensor("indices", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<2>,
            k: Tensor<1, Int>,
        ) -> (Tensor<2>, Tensor<2, Int>) {
            let (values, indices) = {
                let k: usize = {
                    let data = k.to_data().convert::<i64>();
                    data.as_slice::<i64>().unwrap()[0].max(0) as usize
                };
                let (values, indices) = input.topk_with_indices(k, 1);
                (values, indices.cast(burn::tensor::DType::I64))
            };
            (values, indices)
        }
        ");
    }
}
