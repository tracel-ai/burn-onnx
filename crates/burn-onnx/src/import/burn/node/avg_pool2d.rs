use super::prelude::*;

impl NodeCodegen for onnx_ir::node::avg_pool2d::AveragePool2dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let config = &self.config;
        let (top, left, bottom, right) = config.padding.as_tuple();
        let explicit = [(top, bottom), (left, right)];
        let kernel = config.kernel_size;
        let strides = config.strides;
        let dilation = config.dilation;
        let padding = crate::burn::codegen::pool_padding_pairs(
            &config.auto_pad,
            &explicit,
            &self.inputs[0].ty,
            &input,
            &kernel,
            &strides,
            &dilation,
        );
        let kernel = kernel.to_tokens();
        let strides = strides.to_tokens();
        let dilation = dilation.to_tokens();
        let count_include_pad = config.count_include_pad;
        let ceil_mode = config.ceil_mode;
        quote! {
            let #output = {
                let (padding, input) = (#padding, #input);
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new(#kernel)
                        .with_stride(#strides)
                        .with_padding_pairs(padding)
                        .with_dilation(#dilation)
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
    use super::NodeCodegen;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::avg_pool2d::{AveragePool2dNode, AveragePool2dNodeBuilder, AvgPool2dConfig};
    use onnx_ir::padding::{AutoPad, PaddingConfig2d};

    fn create_avg_pool2d_node(name: &str, ceil_mode: bool) -> AveragePool2dNode {
        let config = AvgPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Valid,
            false,
            [1, 1],
            ceil_mode,
            AutoPad::NotSet,
        );

        AveragePool2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    fn create_avg_pool2d_node_asymmetric(name: &str) -> AveragePool2dNode {
        // Asymmetric padding: top=1, left=2, bottom=3, right=4
        let config = AvgPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Explicit(1, 2, 3, 4),
            false,
            [1, 1],
            false,
            AutoPad::NotSet,
        );

        AveragePool2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_avg_pool2d_dilated_explicit() {
        let mut node = create_avg_pool2d_node_asymmetric("pool");
        node.config.dilation = [2; 2];
        node.config.auto_pad = AutoPad::NotSet;
        node.config.count_include_pad = true;
        node.config.ceil_mode = true;
        assert!(node.field().is_none());
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = ([(1usize, 3usize), (2usize, 4usize)], input);
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([2, 2])
                        .with_count_include_pad(true)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool2d_dilated_same_upper() {
        let mut node = create_avg_pool2d_node_asymmetric("pool");
        node.config.dilation = [2; 2];
        node.config.auto_pad = AutoPad::SameUpper;
        node.config.count_include_pad = true;
        node.config.ceil_mode = true;
        assert!(node.field().is_none());
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = (
                    {
                        let dims = input.dims();
                        [
                            {
                                let size = dims[2usize];
                                let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                    + 5usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (small, big)
                            },
                            {
                                let size = dims[3usize];
                                let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                    + 5usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (small, big)
                            },
                        ]
                    },
                    input,
                );
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([2, 2])
                        .with_count_include_pad(true)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool2d_dilated_same_lower() {
        let mut node = create_avg_pool2d_node_asymmetric("pool");
        node.config.dilation = [2; 2];
        node.config.auto_pad = AutoPad::SameLower;
        node.config.count_include_pad = true;
        node.config.ceil_mode = true;
        assert!(node.field().is_none());
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = (
                    {
                        let dims = input.dims();
                        [
                            {
                                let size = dims[2usize];
                                let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                    + 5usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (big, small)
                            },
                            {
                                let size = dims[3usize];
                                let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                    + 5usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (big, small)
                            },
                        ]
                    },
                    input,
                );
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([2, 2])
                        .with_count_include_pad(true)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool2d_forward() {
        let node = create_avg_pool2d_node("pool1", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = ([(0usize, 0usize), (0usize, 0usize)], input);
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_count_include_pad(false)
                        .with_ceil_mode(false),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool2d_forward_with_clone() {
        let node = create_avg_pool2d_node("pool1", false);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = ([(0usize, 0usize), (0usize, 0usize)], input.clone());
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_count_include_pad(false)
                        .with_ceil_mode(false),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool2d_forward_ceil_mode_true() {
        let node = create_avg_pool2d_node("pool1", true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = ([(0usize, 0usize), (0usize, 0usize)], input);
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_count_include_pad(false)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool2d_forward_auto_pad_same_upper() {
        let config = AvgPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Valid,
            false,
            [1, 1],
            false,
            AutoPad::SameUpper,
        );
        let node = AveragePool2dNodeBuilder::new("pool1")
            .input_tensor_shape("input", vec![1, 3, 7, 7], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = ([(1usize, 1usize), (1usize, 1usize)], input);
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_count_include_pad(false)
                        .with_ceil_mode(false),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool2d_forward_auto_pad_same_upper_dynamic() {
        let config = AvgPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Valid,
            false,
            [1, 1],
            false,
            AutoPad::SameUpper,
        );
        let node = AveragePool2dNodeBuilder::new("pool1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = (
                    {
                        let dims = input.dims();
                        [
                            {
                                let size = dims[2usize];
                                let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                    + 3usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (small, big)
                            },
                            {
                                let size = dims[3usize];
                                let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                    + 3usize)
                                    .saturating_sub(size);
                                let small = total / 2;
                                let big = total - small;
                                (small, big)
                            },
                        ]
                    },
                    input,
                );
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_count_include_pad(false)
                        .with_ceil_mode(false),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool2d_forward_asymmetric_padding() {
        let node = create_avg_pool2d_node_asymmetric("pool1");
        let code = codegen_forward_default(&node);
        // Asymmetric padding is passed directly to the functional API
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let (padding, input) = ([(1usize, 3usize), (2usize, 4usize)], input);
                burn::tensor::module::avg_pool2d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3, 3])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_count_include_pad(false)
                        .with_ceil_mode(false),
                )
            };
            output
        }
        ");
    }
}
