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
        let input = scope.arg(self.inputs.first().unwrap());

        // TopK has 2 outputs
        let values_output = arg_to_ident(&self.outputs[0]);
        let indices_output = arg_to_ident(&self.outputs[1]);

        let axis = self.config.axis.to_tokens();

        // Extract static k from the enum wrapper
        let k = match &self.config.k {
            onnx_ir::topk::TopKInput::Static(k_value) => k_value.to_tokens(),
            onnx_ir::topk::TopKInput::Runtime(_) => {
                panic!("Runtime k value is not supported in burn-onnx")
            }
        };

        // Determine behavior based on `largest` flag.  Burn's primitive only
        // selects largest elements and always returns a sorted result.  When
        // lowest values are requested we negate the tensor before and after the
        // call; the `sorted` flag is ignored since sorted output is valid when
        // unsorted was requested.
        if self.config.largest {
            quote! {
                let (#values_output, #indices_output) = #input.topk_with_indices(#k, #axis);
            }
        } else {
            // isolate temporaries inside a block to avoid name collisions when
            // multiple bottom-K nodes appear in the same forward method
            quote! {
                let (#values_output, #indices_output) = {
                    let (v, i) = #input.neg().topk_with_indices(#k, #axis);
                    (v.neg(), i)
                };
            }
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
        pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
            let (values, indices) = input.topk_with_indices(5, 1);
            (values, indices)
        }
        ");
    }

    #[test]
    fn test_top_k_bottom() {
        let config = TopKConfig::new(1, TopKInput::Static(5), false, true);
        let node = TopKNodeBuilder::new("topk1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("values", 2, DType::F32)
            .output_tensor("indices", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
            let (values, indices) = {
                let (v, i) = input.neg().topk_with_indices(5, 1);
                (v.neg(), i)
            };
            (values, indices)
        }
        ");
    }

    #[test]
    fn test_top_k_unsorted_flag() {
        // unsorted flag is ignored; codegen should be identical to default
        let config = TopKConfig::new(1, TopKInput::Static(5), true, false);
        let node = TopKNodeBuilder::new("topk1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("values", 2, DType::F32)
            .output_tensor("indices", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
            let (values, indices) = input.topk_with_indices(5, 1);
            (values, indices)
        }
        ");
    }
}
