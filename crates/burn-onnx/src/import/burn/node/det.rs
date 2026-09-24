use super::prelude::*;

impl NodeCodegen for onnx_ir::node::det::DetNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        let input_rank = match &input_arg.ty {
            ArgType::Tensor(t) => t.rank,
            other => unreachable!("Det input type validated in onnx-ir, got {other:?}"),
        };

        // burn's det works on the last two axes of a batched input of rank >= 3, so a
        // single matrix gets a leading batch axis of 1. Its [1]-shaped result is how
        // a scalar is represented on device.
        if input_rank == 2 {
            quote! {
                let #output = burn::tensor::linalg::det::<3, 2, 1>(#input.unsqueeze_dim(0));
            }
        } else {
            let [rank, rank_minus_one, rank_minus_two] =
                [input_rank, input_rank - 1, input_rank - 2].map(|r| r.to_tokens());
            quote! {
                let #output = burn::tensor::linalg::det::<#rank, #rank_minus_one, #rank_minus_two>(#input);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::det::DetNodeBuilder;

    #[test]
    fn test_det_2d_forward() {
        let node = DetNodeBuilder::new("det1")
            .input_tensor("input", 2, DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<1> {
            let output = burn::tensor::linalg::det::<3, 2, 1>(input.unsqueeze_dim(0));
            output
        }
        ");
    }

    #[test]
    fn test_det_3d_forward() {
        let node = DetNodeBuilder::new("det1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<1> {
            let output = burn::tensor::linalg::det::<3, 2, 1>(input);
            output
        }
        ");
    }

    #[test]
    fn test_det_4d_forward() {
        let node = DetNodeBuilder::new("det1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<2> {
            let output = burn::tensor::linalg::det::<4, 3, 2>(input);
            output
        }
        ");
    }
}
