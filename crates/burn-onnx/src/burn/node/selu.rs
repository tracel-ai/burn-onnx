use super::prelude::*;

const DEFAULT_ALPHA: f64 = 1.673_263_192_176_818_8;
const DEFAULT_GAMMA: f64 = 1.050_701_022_148_132_3;

impl NodeCodegen for onnx_ir::selu::SeluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // Burn's selu() uses hardcoded alpha/gamma matching the ONNX defaults.
        // Reject non-default values since Burn doesn't support custom parameters.
        if (self.config.alpha - DEFAULT_ALPHA).abs() > 1e-6 {
            panic!(
                "Selu: custom alpha ({}) is not supported by Burn, only the default ({}) is supported",
                self.config.alpha, DEFAULT_ALPHA
            );
        }
        if (self.config.gamma - DEFAULT_GAMMA).abs() > 1e-6 {
            panic!(
                "Selu: custom gamma ({}) is not supported by Burn, only the default ({}) is supported",
                self.config.gamma, DEFAULT_GAMMA
            );
        }

        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = burn::tensor::activation::selu(#input);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::selu::{SeluConfig, SeluNode, SeluNodeBuilder};

    fn create_node(name: &str) -> SeluNode {
        let config = SeluConfig::new(1.67326319217681884765625, 1.05070102214813232421875);

        SeluNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_selu_forward() {
        let node = create_node("selu1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::selu(input);
            output
        }
        ");
    }

    #[test]
    #[should_panic(expected = "custom alpha")]
    fn test_selu_rejects_custom_alpha() {
        let config = SeluConfig::new(2.0, 1.05070102214813232421875);
        let node = SeluNodeBuilder::new("selu1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        codegen_forward_default(&node);
    }

    #[test]
    #[should_panic(expected = "custom gamma")]
    fn test_selu_rejects_custom_gamma() {
        let config = SeluConfig::new(1.67326319217681884765625, 2.0);
        let node = SeluNodeBuilder::new("selu1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        codegen_forward_default(&node);
    }
}
