use super::prelude::*;

impl NodeCodegen for onnx_ir::size::SizeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        // ONNX Size is the element count of the input. The output is always
        // ScalarNative(I64), so every branch has to yield an i64.
        //
        // The branches below answer from the type alone, but each still
        // takes its input through `scope.arg` and binds it. `scope.arg`
        // is what decrements the clone-tracking refcount, so skipping it
        // for a ScalarTensor would leave a phantom reference and make
        // some later consumer emit a clone it does not need. Binding the
        // result keeps the generated forward free of unused-variable
        // warnings when Size is the input's only consumer.
        let count = match &input_arg.ty {
            ArgType::Tensor(_) => {
                let input = scope.arg(input_arg);
                quote! { #input.shape().num_elements() as i64 }
            }
            // A Shape(N) is a native [i64; N] array standing in for a 1-D
            // int64 tensor, so its element count is known at codegen time.
            ArgType::Shape(rank) => {
                let input = scope.arg(input_arg);
                let count = *rank as i64;
                quote! { { let _ = &#input; #count } }
            }
            // Both scalar forms hold exactly one element.
            ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => {
                let input = scope.arg(input_arg);
                quote! { { let _ = &#input; 1i64 } }
            }
        };

        quote! {
            let #output = #count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::size::SizeNodeBuilder;

    #[test]
    fn test_size_forward() {
        let node = SizeNodeBuilder::new("size1")
            .input_tensor("input", 2, DType::F32)
            .output_scalar("output", DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> i64 {
            let output = input.shape().num_elements() as i64;
            output
        }
        ");
    }

    #[test]
    fn test_size_forward_shape_input() {
        let node = SizeNodeBuilder::new("size1")
            .input_shape("input", 3)
            .output_scalar("output", DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: [i64; 3]) -> i64 {
            let output = {
                let _ = &input;
                3i64
            };
            output
        }
        ");
    }

    #[test]
    fn test_size_forward_scalar_input() {
        let node = SizeNodeBuilder::new("size1")
            .input_scalar("input", DType::F32)
            .output_scalar("output", DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> i64 {
            let output = {
                let _ = &input;
                1i64
            };
            output
        }
        ");
    }
}
