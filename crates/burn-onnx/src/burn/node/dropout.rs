use super::prelude::*;

impl NodeCodegen for onnx_ir::dropout::DropoutNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // Dropout is an identity in inference mode (the only mode burn-onnx
        // generates). onnx-ir's DropoutProcessor declares `is_noop = true`
        // for the single-output case, which lets the noop-elimination pass
        // remove the node when graph simplification runs. This codegen
        // fires in the cases where that pass can't drop the node:
        //
        //   1. Simplification is disabled (e.g. `onnx2burn --no-simplify`).
        //   2. The optional `mask` output is present. Noop elimination only
        //      rewires `output[0]`, so dropping the node would orphan the
        //      mask; `DropoutProcessor::is_noop` returns false in that case
        //      and we reach this path.
        //
        // Either way we ignore the ratio and training_mode inputs (static
        // or runtime) and emit the `y` output as an alias of `input`. When
        // the mask output is requested, we also emit an all-true Bool
        // tensor of the same shape (every element survives, which is what
        // training_mode=false yields).
        let input_arg = self.inputs.first().unwrap();
        let input = scope.arg(input_arg);
        let y = arg_to_ident(self.outputs.first().unwrap());

        if self.outputs.len() == 1 {
            return quote! {
                let #y = #input;
            };
        }

        let mask = arg_to_ident(self.outputs.get(1).unwrap());
        let rank = input_arg.ty.rank();
        quote! {
            let #y = #input;
            let #mask = {
                let __dropout_ones: Tensor<B, #rank, burn::tensor::Int> =
                    Tensor::ones(#y.shape(), &self.device);
                __dropout_ones.not_equal_elem(0i64)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::dropout::{DropoutConfig, DropoutInput, DropoutNodeBuilder};

    #[test]
    fn test_dropout_forward_static() {
        let config = DropoutConfig::new(DropoutInput::Static(0.5));
        let node = DropoutNodeBuilder::new("dropout1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        // Inference-only: dropout is identity regardless of the static ratio.
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input;
            output
        }
        ");
    }

    #[test]
    fn test_dropout_forward_with_mask_output() {
        // Opset 12+ Dropout can emit a mask output. In inference mode the
        // mask is all-true (every element survives).
        let config = DropoutConfig::new(DropoutInput::Static(0.5));
        let node = DropoutNodeBuilder::new("dropout1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .output_tensor(
                "mask",
                2,
                burn::tensor::DType::Bool(burn::tensor::BoolStore::Native),
            )
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2, Bool>) {
            let output = input;
            let mask = {
                let __dropout_ones: Tensor<B, 2usize, burn::tensor::Int> = Tensor::ones(
                    output.shape(),
                    &self.device,
                );
                __dropout_ones.not_equal_elem(0i64)
            };
            (output, mask)
        }
        ");
    }

    #[test]
    fn test_dropout_forward_runtime_ratio() {
        // Opset 12+ gives Dropout a runtime ratio input. We still emit
        // identity in inference mode, so codegen should not panic and
        // should ignore the runtime input entirely.
        let config = DropoutConfig::new(DropoutInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
            "ratio".to_string(),
            1,
        )));
        let node = DropoutNodeBuilder::new("dropout1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar("ratio", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, ratio: f32) -> Tensor<B, 2> {
            let output = input;
            output
        }
        ");
    }
}
