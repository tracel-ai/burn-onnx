use super::prelude::*;

impl NodeCodegen for onnx_ir::trilu::TriluNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let input = scope.arg(input_arg);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let diagonal = self.config.diagonal.to_tokens();

        // burn-flex's Bool tensor doesn't satisfy the trait bounds for tril/triu,
        // so round-trip Bool through Int.
        let is_bool = matches!(&input_arg.ty, ArgType::Tensor(t) if t.dtype.is_bool());

        let body = if self.config.upper {
            quote! { triu(#diagonal) }
        } else {
            quote! { tril(#diagonal) }
        };

        if is_bool {
            quote! {
                let #output = #input.int().#body.bool();
            }
        } else {
            quote! {
                let #output = #input.#body;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::trilu::{TriluConfig, TriluNodeBuilder};

    #[test]
    fn test_trilu_upper() {
        let config = TriluConfig::new(true, 0);
        let node = TriluNodeBuilder::new("triu1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input.triu(0);
            output
        }
        ");
    }

    #[test]
    fn test_trilu_lower() {
        let config = TriluConfig::new(false, 1);
        let node = TriluNodeBuilder::new("tril1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input.tril(1);
            output
        }
        ");
    }

    #[test]
    fn test_trilu_bool_input_lower() {
        let config = TriluConfig::new(false, 0);
        let node = TriluNodeBuilder::new("tril1")
            .input_tensor("mask", 2, DType::Bool(BoolStore::Native))
            .output_tensor("masked", 2, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, mask: Tensor<2, Bool>) -> Tensor<2, Bool> {
            let masked = mask.int().tril(0).bool();
            masked
        }
        ");
    }

    #[test]
    fn test_trilu_bool_input_upper() {
        let config = TriluConfig::new(true, -1);
        let node = TriluNodeBuilder::new("triu1")
            .input_tensor("mask", 3, DType::Bool(BoolStore::Native))
            .output_tensor("masked", 3, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, mask: Tensor<3, Bool>) -> Tensor<3, Bool> {
            let masked = mask.int().triu(-1).bool();
            masked
        }
        ");
    }
}
