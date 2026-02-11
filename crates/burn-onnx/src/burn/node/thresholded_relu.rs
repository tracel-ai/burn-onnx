use super::prelude::*;

impl NodeCodegen for onnx_ir::thresholded_relu::ThresholdedReluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let alpha = self.config.alpha.to_tokens();

        quote! {
            let #output = burn::tensor::activation::thresholded_relu(#input, #alpha);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::thresholded_relu::{
        ThresholdedReluConfig, ThresholdedReluNode, ThresholdedReluNodeBuilder,
    };

    fn create_node(name: &str, alpha: f64) -> ThresholdedReluNode {
        let config = ThresholdedReluConfig::new(alpha);

        ThresholdedReluNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_thresholded_relu_forward_default_alpha() {
        let node = create_node("thresholded_relu1", 1.0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::thresholded_relu(input, 1.0);
            output
        }
        ");
    }

    #[test]
    fn test_thresholded_relu_forward_custom_alpha() {
        let node = create_node("thresholded_relu1", 0.5);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::thresholded_relu(input, 0.5);
            output
        }
        ");
    }
}
