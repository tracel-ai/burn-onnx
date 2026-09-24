use super::prelude::*;
use onnx_ir::node::gelu::GeluApproximate;

impl NodeCodegen for onnx_ir::node::gelu::GeluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let function = match self.config.approximate {
            GeluApproximate::Exact => quote! { gelu },
            GeluApproximate::Tanh => quote! { gelu_approximate },
        };

        quote! {
            let #output = burn::tensor::activation::#function(#input);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::gelu::{GeluApproximate, GeluConfig, GeluNodeBuilder};

    fn code_for(approximate: GeluApproximate) -> String {
        let node = GeluNodeBuilder::new("gelu1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(GeluConfig { approximate })
            .build();
        codegen_forward_default(&node)
    }

    #[test]
    fn test_gelu_forward() {
        assert_snapshot!(code_for(GeluApproximate::Exact), @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = burn::tensor::activation::gelu(input);
            output
        }
        ");
    }

    #[test]
    fn test_gelu_tanh_forward() {
        assert_snapshot!(code_for(GeluApproximate::Tanh), @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = burn::tensor::activation::gelu_approximate(input);
            output
        }
        ");
    }
}
