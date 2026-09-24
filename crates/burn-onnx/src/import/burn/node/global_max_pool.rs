use super::prelude::*;

impl NodeCodegen for onnx_ir::node::global_max_pool::GlobalMaxPoolNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // onnx-ir guarantees rank >= 3. The spatial axes reduce to size 1, keeping the
        // [N, C, 1, 1, ...] shape.
        let rank = self.inputs[0].ty.rank();
        let dims = (2..rank).collect::<Vec<usize>>().to_tokens();

        quote! {
            let #output = #input.max_dims(&#dims);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::global_max_pool::GlobalMaxPoolNodeBuilder;

    #[test]
    fn test_global_max_pool_forward() {
        let node = GlobalMaxPoolNodeBuilder::new("global_max_pool1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = input.max_dims(&[2, 3]);
            output
        }
        ");
    }
}
