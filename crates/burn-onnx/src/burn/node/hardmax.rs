use super::prelude::*;

impl NodeCodegen for onnx_ir::hardmax::HardmaxNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let dim = self.config.axis.to_tokens();

        quote! {
            let #output = {
                let input = #input;
                let indices = input.clone().argmax(#dim);
                input.zeros_like().scatter(
                    #dim,
                    indices.clone(),
                    Tensor::ones(indices.dims(), &input.device()),
                    burn::tensor::IndexingUpdateOp::Add,
                )
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::hardmax::{HardmaxConfig, HardmaxNode, HardmaxNodeBuilder};

    fn create_hardmax_node(name: &str, rank: usize, axis: usize) -> HardmaxNode {
        let config = HardmaxConfig::new(axis);

        HardmaxNodeBuilder::new(name)
            .input_tensor("input", rank, DType::F32)
            .output_tensor("output", rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_hardmax_forward_last_axis() {
        let node = create_hardmax_node("hardmax1", 2, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let input = input;
                let indices = input.clone().argmax(1);
                input
                    .zeros_like()
                    .scatter(
                        1,
                        indices.clone(),
                        Tensor::ones(indices.dims(), &input.device()),
                        burn::tensor::IndexingUpdateOp::Add,
                    )
            };
            output
        }
        ");
    }

    #[test]
    fn test_hardmax_forward_first_axis() {
        let node = create_hardmax_node("hardmax1", 3, 0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let input = input;
                let indices = input.clone().argmax(0);
                input
                    .zeros_like()
                    .scatter(
                        0,
                        indices.clone(),
                        Tensor::ones(indices.dims(), &input.device()),
                        burn::tensor::IndexingUpdateOp::Add,
                    )
            };
            output
        }
        ");
    }
}
