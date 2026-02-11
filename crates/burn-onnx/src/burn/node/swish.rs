use super::prelude::*;

impl NodeCodegen for onnx_ir::swish::SwishNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        if self.config.alpha == 1.0 {
            quote! {
                let #output = burn::tensor::activation::silu(#input);
            }
        } else {
            let alpha = self.config.alpha.to_tokens();
            quote! {
                let #output = #input.clone() * burn::tensor::activation::sigmoid(#input * #alpha);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::swish::{SwishConfig, SwishNode, SwishNodeBuilder};

    fn create_swish_node(name: &str, alpha: f64) -> SwishNode {
        let config = SwishConfig::new(alpha);

        SwishNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_swish_forward_default_alpha() {
        let node = create_swish_node("swish1", 1.0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::silu(input);
            output
        }
        ");
    }

    #[test]
    fn test_swish_forward_custom_alpha() {
        let node = create_swish_node("swish1", 0.5);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clone() * burn::tensor::activation::sigmoid(input * 0.5);
            output
        }
        ");
    }
}
