use super::prelude::*;
use onnx_ir::node::padding::PaddingConfig3d;

/// `(begin, end)` padding per spatial dim (depth, height, width) for the pool options:
/// the explicit pads, resolved `auto_pad` pads when the input size is static, or the pads
/// computed at run time from the input size for `SAME_*` on a dynamic input.
pub(super) fn pool3d_padding(
    padding: &PaddingConfig3d,
    auto_pad: &onnx_ir::node::padding::AutoPad,
    input_ty: &ArgType,
    input: &TokenStream,
    kernel: &[usize; 3],
    strides: &[usize; 3],
    dilation: &[usize; 3],
) -> TokenStream {
    let explicit = match padding {
        PaddingConfig3d::Valid => [(0, 0); 3],
        PaddingConfig3d::Explicit(front, top, left, back, bottom, right) => {
            [(*front, *back), (*top, *bottom), (*left, *right)]
        }
    };
    let input_spatial = onnx_ir::node::padding::static_spatial_dims(input_ty);
    match crate::burn::codegen::resolve_padding_pairs(
        auto_pad,
        &explicit,
        input_spatial.as_deref(),
        kernel,
        strides,
        dilation,
    ) {
        Some(pairs) => crate::burn::codegen::padding_pairs_tokens(&pairs),
        None => {
            crate::burn::codegen::runtime_same_padding(auto_pad, input, kernel, strides, dilation)
        }
    }
}

impl NodeCodegen for onnx_ir::node::avg_pool3d::AveragePool3dNode {
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
        let config = &self.config;

        // burn's avg_pool3d has no dilation; onnx-ir rejects dilations other than 1.
        let padding = pool3d_padding(
            &config.padding,
            &config.auto_pad,
            &input_arg.ty,
            &input,
            &config.kernel_size,
            &config.strides,
            &config.dilation,
        );
        let kernel = config.kernel_size.to_tokens();
        let strides = config.strides.to_tokens();
        let count_include_pad = config.count_include_pad;
        let ceil_mode = config.ceil_mode;

        quote! {
            let #output = {
                // The padding may read the input's dims, so it's computed before the input
                // moves; both graph values are read before any temporary is declared.
                let (padding, input) = (#padding, #input);
                burn::tensor::module::avg_pool3d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new(#kernel)
                        .with_stride(#strides)
                        .with_padding_pairs(padding)
                        .with_count_include_pad(#count_include_pad)
                        .with_ceil_mode(#ceil_mode),
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
    use onnx_ir::node::avg_pool3d::{AveragePool3dNode, AveragePool3dNodeBuilder, AvgPool3dConfig};
    use onnx_ir::node::padding::{AutoPad, PaddingConfig3d};

    fn node(input: &str, padding: PaddingConfig3d, auto_pad: AutoPad) -> AveragePool3dNode {
        let config = AvgPool3dConfig::new(
            [2, 3, 2],
            [2, 2, 2],
            padding,
            false,
            [1, 1, 1],
            true,
            auto_pad,
        );
        AveragePool3dNodeBuilder::new("avgpool1")
            .input_tensor(input, 5, DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_avg_pool3d_explicit_padding() {
        let node = node(
            "input",
            PaddingConfig3d::Explicit(1, 0, 1, 0, 1, 0),
            AutoPad::NotSet,
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = {
                let (padding, input) = (
                    [(1usize, 0usize), (0usize, 1usize), (1usize, 0usize)],
                    input,
                );
                burn::tensor::module::avg_pool3d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([2, 3, 2])
                        .with_stride([2, 2, 2])
                        .with_padding_pairs(padding)
                        .with_count_include_pad(false)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool3d_same_lower_runtime_padding() {
        // The input has no static shape, so SAME_LOWER's pads are computed at run time
        let node = node("input", PaddingConfig3d::Valid, AutoPad::SameLower);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = {
                let (padding, input) = (
                    {
                        let dims = input.dims();
                        [
                            {
                                let size = dims[2usize];
                                let total = (size.div_ceil(2usize).saturating_sub(1) * 2usize
                                    + 2usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (big, small)
                            },
                            {
                                let size = dims[3usize];
                                let total = (size.div_ceil(2usize).saturating_sub(1) * 2usize
                                    + 3usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (big, small)
                            },
                            {
                                let size = dims[4usize];
                                let total = (size.div_ceil(2usize).saturating_sub(1) * 2usize
                                    + 2usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (big, small)
                            },
                        ]
                    },
                    input,
                );
                burn::tensor::module::avg_pool3d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([2, 3, 2])
                        .with_stride([2, 2, 2])
                        .with_padding_pairs(padding)
                        .with_count_include_pad(false)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn input_named_like_its_temporaries_is_accepted() {
        for temporary in ["padding", "input", "dims", "size", "total", "small", "big"] {
            for auto_pad in [AutoPad::NotSet, AutoPad::SameLower] {
                let node = node(temporary, PaddingConfig3d::Valid, auto_pad);
                assert_eq!(
                    shadow_check_result(&node),
                    Ok(()),
                    "input named {temporary}"
                );
            }
        }
    }
}
