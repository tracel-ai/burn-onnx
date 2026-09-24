use super::prelude::*;
use onnx_ir::depth_to_space::DepthToSpaceMode;

impl NodeCodegen for onnx_ir::depth_to_space::DepthToSpaceNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let block_size = self.config.block_size;

        // burn's PixelShuffle takes float tensors only; other kinds are rearranged
        // directly.
        if !self.inputs[0].ty.elem_type().is_float() {
            let (split, permutation) = match self.config.mode {
                DepthToSpaceMode::Dcr => (
                    quote! { [b, #block_size, #block_size, c / (#block_size * #block_size), h, w] },
                    quote! { [0, 3, 4, 1, 5, 2] },
                ),
                DepthToSpaceMode::Crd => (
                    quote! { [b, c / (#block_size * #block_size), #block_size, #block_size, h, w] },
                    quote! { [0, 1, 4, 2, 5, 3] },
                ),
            };
            return quote! {
                let #output = {
                    let [b, c, h, w] = #input.dims();
                    #input
                        .reshape(#split)
                        .permute(#permutation)
                        .reshape([b, c / (#block_size * #block_size), h * #block_size, w * #block_size])
                };
            };
        }

        let shuffle = quote! {
            burn::nn::PixelShuffleConfig::new(#block_size).init()
        };

        // burn's PixelShuffle is the CRD layout: output channel c takes input channels
        // c * b^2 .. (c + 1) * b^2. DCR orders input channels as (block_h, block_w, c),
        // so they are regrouped into CRD's (c, block_h, block_w) first.
        match self.config.mode {
            DepthToSpaceMode::Crd => quote! {
                let #output = #shuffle.forward(#input);
            },
            DepthToSpaceMode::Dcr => quote! {
                let #output = {
                    let [b, c, h, w] = #input.dims();
                    let crd = #input
                        .reshape([b, #block_size * #block_size, c / (#block_size * #block_size), h, w])
                        .swap_dims(1, 2)
                        .reshape([b, c, h, w]);
                    #shuffle.forward(crd)
                };
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::depth_to_space::{DepthToSpaceConfig, DepthToSpaceMode, DepthToSpaceNodeBuilder};

    #[test]
    fn test_depth_to_space_dcr() {
        let config = DepthToSpaceConfig {
            block_size: 2,
            mode: DepthToSpaceMode::Dcr,
        };
        let node = DepthToSpaceNodeBuilder::new("d2s1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let [b, c, h, w] = input.dims();
                let crd = input
                    .reshape([b, 2usize * 2usize, c / (2usize * 2usize), h, w])
                    .swap_dims(1, 2)
                    .reshape([b, c, h, w]);
                burn::nn::PixelShuffleConfig::new(2usize).init().forward(crd)
            };
            output
        }
        ");
    }

    #[test]
    fn test_depth_to_space_crd() {
        let config = DepthToSpaceConfig {
            block_size: 2,
            mode: DepthToSpaceMode::Crd,
        };
        let node = DepthToSpaceNodeBuilder::new("d2s2")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = burn::nn::PixelShuffleConfig::new(2usize).init().forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_int_input() {
        let node = DepthToSpaceNodeBuilder::new("rearrange1")
            .input_tensor("input", 4, DType::I64)
            .output_tensor("output", 4, DType::I64)
            .config(DepthToSpaceConfig::new(DepthToSpaceMode::Dcr, 2))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4, Int>) -> Tensor<4, Int> {
            let output = {
                let [b, c, h, w] = input.dims();
                input
                    .reshape([b, 2usize, 2usize, c / (2usize * 2usize), h, w])
                    .permute([0, 3, 4, 1, 5, 2])
                    .reshape([b, c / (2usize * 2usize), h * 2usize, w * 2usize])
            };
            output
        }
        ");
    }
}
