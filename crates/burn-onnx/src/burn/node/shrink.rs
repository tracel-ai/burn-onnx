use super::prelude::*;

impl NodeCodegen for onnx_ir::shrink::ShrinkNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let lambda = self.config.lambda.to_tokens();
        let bias = self.config.bias.to_tokens();

        quote! {
        let #output = burn::tensor::activation::shrink(#input, #lambda, #bias);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::shrink::{ShrinkConfig, ShrinkNode, ShrinkNodeBuilder};

    fn create_shrink_node(name: &str, lambda: f64, bias: f64) -> ShrinkNode {
        let config = ShrinkConfig::new(lambda, bias);

        ShrinkNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_shrink_forward_default_lambda_bias() {
        let node = create_shrink_node("shrink1", 0.0, 0.0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::shrink(input, 0.0, 0.0);
            output
        }
        ");
    }

    #[test]
    fn test_shrink_forward_custom_lambda_bias() {
        let node = create_shrink_node("shrink2", 1.0, 0.5);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::shrink(input, 1.0, 0.5);
            output
        }
        ");
    }
}
