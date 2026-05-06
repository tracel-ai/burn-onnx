use super::prelude::*;

impl NodeCodegen for onnx_ir::node::not::NotNode {
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

        match &input_arg.ty {
            // Shape arguments are host-side `[i64; N]` arrays whose elements encode
            // booleans as 0/1 (mirrors the Equal-on-Shape codegen). `.bool_not()`
            // is not defined on a plain array, so flip in place.
            ArgType::Shape(_) => quote! {
                let #output = {
                    let mut result = #input;
                    for v in result.iter_mut() {
                        *v = if *v != 0 { 0i64 } else { 1i64 };
                    }
                    result
                };
            },
            // Native host scalar — emit the Rust `!` operator. `.bool_not()` would
            // not compile against a `bool` binding.
            ArgType::ScalarNative(_) => quote! {
                let #output = !#input;
            },
            // On-device tensors (Tensor and ScalarTensor) carry `.bool_not()`.
            ArgType::Tensor(_) | ArgType::ScalarTensor(_) => quote! {
                let #output = #input.bool_not();
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::node::not::NotNodeBuilder;

    #[test]
    fn test_not_forward() {
        let node = NotNodeBuilder::new("not1")
            .input_tensor("input", 2, DType::Bool(BoolStore::Native))
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
            let output = input.bool_not();
            output
        }
        ");
    }

    #[test]
    fn test_not_shape_input() {
        let node = NotNodeBuilder::new("not1")
            .input_shape("flags", 3)
            .output_shape("inverted", 3)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, flags: [i64; 3]) -> [i64; 3] {
            let inverted = {
                let mut result = flags;
                for v in result.iter_mut() {
                    *v = if *v != 0 { 0i64 } else { 1i64 };
                }
                result
            };
            inverted
        }
        ");
    }
}
