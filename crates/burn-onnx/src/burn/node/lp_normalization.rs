use super::prelude::*;
use onnx_ir::node::lp_normalization::LpNormalizationNode;

impl NodeCodegen for LpNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let axis = self.config.axis.to_tokens();

        // Relies on l1_norm/l2_norm reducing `axis` to size 1 so the final
        // division broadcasts against x.
        let norm_expr = match self.config.p {
            1 => quote! { burn::tensor::linalg::l1_norm(x.clone(), #axis) },
            2 => quote! { burn::tensor::linalg::l2_norm(x.clone(), #axis) },
            p => unreachable!("p must be 1 or 2, got {p}"),
        };

        quote! {
            let #output = {
                let x = #input;
                let norm = #norm_expr;
                x / norm
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::lp_normalization::{LpNormalizationConfig, LpNormalizationNodeBuilder};

    #[test]
    fn l2_default_last_axis() {
        let node = LpNormalizationNodeBuilder::new("lpnorm1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(LpNormalizationConfig::new(2, 2))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input;
                let norm = burn::tensor::linalg::l2_norm(x.clone(), 2);
                x / norm
            };
            output
        }
        ");
    }

    #[test]
    fn l1_along_axis_0() {
        let node = LpNormalizationNodeBuilder::new("lpnorm1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(LpNormalizationConfig::new(0, 1))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let x = input;
                let norm = burn::tensor::linalg::l1_norm(x.clone(), 0);
                x / norm
            };
            output
        }
        ");
    }

    #[test]
    fn l2_middle_axis() {
        let node = LpNormalizationNodeBuilder::new("lpnorm1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(LpNormalizationConfig::new(1, 2))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let x = input;
                let norm = burn::tensor::linalg::l2_norm(x.clone(), 1);
                x / norm
            };
            output
        }
        ");
    }

    #[test]
    fn forward_with_clone() {
        let node = LpNormalizationNodeBuilder::new("lpnorm1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(LpNormalizationConfig::new(2, 2))
            .build();
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input.clone();
                let norm = burn::tensor::linalg::l2_norm(x.clone(), 2);
                x / norm
            };
            output
        }
        ");
    }
}
