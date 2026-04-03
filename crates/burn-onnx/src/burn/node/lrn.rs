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

        // NOTE: `input`'s ArgType is already checked in `onnx-ir`

        // TODO: Re-implement using vectorized conv ops natively in Burn
        // See https://github.com/tracel-ai/burn/issues/4724.
        quote! {
            let shape = #input.dims(); // [N, C, D1, ..., Dk]
            let num_channels = shape[1];

            // NOTE: The window width is asymmetric for
            // even `size`: pad_left = floor((size-1)/2), pad_right = ceil((size-1)/2) = size/2.
            let pad_left = (#size - 1) / 2;
            let pad_right = #size / 2;

            // Compute `square_sum` by accumulating channel-by-channel.
            let squared = #input.clone().square();
            let mut square_sum = Tensor::zeros_like(&squared);
            for c in 0..num_channels {
                let win_start = c.saturating_sub(pad_left).max(0);
                let win_end = (c + pad_right + 1).min(num_channels); // exclusive
                let win_len = win_end - win_start;
                // Sum over the channel window [win_start, win_end) using narrow
                let window_sum = squared.clone().narrow(1, win_start, win_len).sum_dim(1);
                square_sum = square_sum.slice_assign([0..shape[0], c..c + 1], window_sum);
            }

            let #output = #input / (#bias + #alpha / #size as f32 * square_sum).powf_scalar(#beta);
        }
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
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let shape = input.dims();
            let num_channels = shape[1];
            let pad_left = (2 - 1) / 2;
            let pad_right = 2 / 2;
            let squared = input.clone().square();
            let mut square_sum = Tensor::zeros_like(&squared);
            for c in 0..num_channels {
                let win_start = c.saturating_sub(pad_left).max(0);
                let win_end = (c + pad_right + 1).min(num_channels);
                let win_len = win_end - win_start;
                let window_sum = squared.clone().narrow(1, win_start, win_len).sum_dim(1);
                square_sum = square_sum.slice_assign([0..shape[0], c..c + 1], window_sum);
            }
            let output = input / (1f32 + 0.0001f32 / 2 as f32 * square_sum).powf_scalar(0.75f32);
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
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let shape = input.dims();
            let num_channels = shape[1];
            let pad_left = (10 - 1) / 2;
            let pad_right = 10 / 2;
            let squared = input.clone().square();
            let mut square_sum = Tensor::zeros_like(&squared);
            for c in 0..num_channels {
                let win_start = c.saturating_sub(pad_left).max(0);
                let win_end = (c + pad_right + 1).min(num_channels);
                let win_len = win_end - win_start;
                let window_sum = squared.clone().narrow(1, win_start, win_len).sum_dim(1);
                square_sum = square_sum.slice_assign([0..shape[0], c..c + 1], window_sum);
            }
            let output = input / (2f32 + 0.0002f32 / 10 as f32 * square_sum).powf_scalar(0.5f32);
            output
        }
        ");
    }
}
