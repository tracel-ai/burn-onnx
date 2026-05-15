use super::prelude::*;
use crate::burn::codegen::f32_to_tokens;
use onnx_ir::node::lrn::LrnNode;

impl NodeCodegen for LrnNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let alpha = f32_to_tokens(self.config.alpha);
        let beta = f32_to_tokens(self.config.beta);
        let bias = f32_to_tokens(self.config.bias);
        let size = self.config.size.to_tokens();

        quote! {
            let #output = LocalResponseNormConfig::new(#size as usize)
                .with_alpha(f64::from(#alpha))
                .with_beta(f64::from(#beta))
                .with_k(f64::from(#bias))
                .init()
                .forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::LocalResponseNormConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::lrn::{LrnConfig, LrnNodeBuilder};

    #[test]
    fn test_lrn_forward_default_params() {
        let config = LrnConfig::new(0.0001, 0.75, 1.0, 2);
        let node = LrnNodeBuilder::new("lrn")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = LocalResponseNormConfig::new(2 as usize)
                .with_alpha(f64::from(0.0001f32))
                .with_beta(f64::from(0.75f32))
                .with_k(f64::from(1f32))
                .init()
                .forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_lrn_forward_custom_params() {
        let config = LrnConfig::new(0.0002, 0.5, 2.0, 10);
        let node = LrnNodeBuilder::new("lrn")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = LocalResponseNormConfig::new(10 as usize)
                .with_alpha(f64::from(0.0002f32))
                .with_beta(f64::from(0.5f32))
                .with_k(f64::from(2f32))
                .init()
                .forward(input);
            output
        }
        ");
    }
}
