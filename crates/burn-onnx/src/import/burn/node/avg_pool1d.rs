use super::prelude::*;

impl NodeCodegen for onnx_ir::node::avg_pool1d::AveragePool1dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if self.config.dilation != 1 {
            return None;
        }
        let name = Ident::new(&self.name, Span::call_site());
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.stride.to_tokens();
        let count_include_pad = self.config.count_include_pad;
        let ceil_mode = self.config.ceil_mode;

        let input_spatial = onnx_ir::node::padding::static_spatial_dims(&self.inputs[0].ty);
        let padding = crate::burn::codegen::resolve_auto_pad_1d(
            &self.config.auto_pad,
            &self.config.padding,
            input_spatial.as_deref(),
            self.config.kernel_size,
            self.config.stride,
            self.config.dilation,
        );

        Some(Field::new(
            self.name.clone(),
            quote! {
                AvgPool1d
            },
            quote! {
                let #name = AvgPool1dConfig::new(#kernel_size)
                    .with_stride(#strides)
                    .with_padding(#padding)
                    .with_count_include_pad(#count_include_pad)
                    .with_ceil_mode(#ceil_mode)
                    .init();
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        if self.config.dilation != 1 {
            let config = &self.config;
            let explicit = [config.padding.as_tuple()];
            let kernel = [config.kernel_size];
            let strides = [config.stride];
            let dilation = [config.dilation];
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
            return quote! {
                let #output = {
                    let (padding, input) = (#padding, #input);
                    burn::tensor::module::avg_pool1d(input,
                        burn::tensor::ops::AvgPoolOptions::new(#kernel)
                            .with_stride(#strides).with_padding_pairs(padding)
                            .with_dilation(#dilation).with_count_include_pad(#count_include_pad)
                            .with_ceil_mode(#ceil_mode))
                };
            };
        }
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if self.config.dilation != 1 {
            return;
        }
        imports.register("burn::nn::pool::AvgPool1d");
        imports.register("burn::nn::pool::AvgPool1dConfig");
        imports.register("burn::nn::PaddingConfig1d");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::NodeCodegen;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::avg_pool1d::{AveragePool1dNode, AveragePool1dNodeBuilder, AvgPool1dConfig};
    use onnx_ir::padding::{AutoPad, PaddingConfig1d};

    fn create_avg_pool1d_node(name: &str, ceil_mode: bool) -> AveragePool1dNode {
        let config = AvgPool1dConfig::new(
            3,
            1,
            PaddingConfig1d::Valid,
            false,
            1,
            ceil_mode,
            AutoPad::NotSet,
        );

        AveragePool1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    fn create_avg_pool1d_node_asymmetric(name: &str) -> AveragePool1dNode {
        // Asymmetric padding: left=1, right=2
        let config = AvgPool1dConfig::new(
            3,
            1,
            PaddingConfig1d::Explicit(1, 2),
            false,
            1,
            false,
            AutoPad::NotSet,
        );

        AveragePool1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_avg_pool1d_dilated_explicit() {
        let mut node = create_avg_pool1d_node_asymmetric("pool");
        node.config.dilation = 2;
        node.config.auto_pad = AutoPad::NotSet;
        node.config.count_include_pad = true;
        node.config.ceil_mode = true;
        assert!(node.field().is_none());
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let (padding, input) = ([(1usize, 2usize)], input);
                burn::tensor::module::avg_pool1d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3])
                        .with_stride([1])
                        .with_padding_pairs(padding)
                        .with_dilation([2])
                        .with_count_include_pad(true)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool1d_dilated_same_upper() {
        let mut node = create_avg_pool1d_node_asymmetric("pool");
        node.config.dilation = 2;
        node.config.auto_pad = AutoPad::SameUpper;
        node.config.count_include_pad = true;
        node.config.ceil_mode = true;
        assert!(node.field().is_none());
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
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
                        ]
                    },
                    input,
                );
                burn::tensor::module::avg_pool1d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3])
                        .with_stride([1])
                        .with_padding_pairs(padding)
                        .with_dilation([2])
                        .with_count_include_pad(true)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool1d_dilated_same_lower() {
        let mut node = create_avg_pool1d_node_asymmetric("pool");
        node.config.dilation = 2;
        node.config.auto_pad = AutoPad::SameLower;
        node.config.count_include_pad = true;
        node.config.ceil_mode = true;
        assert!(node.field().is_none());
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
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
                        ]
                    },
                    input,
                );
                burn::tensor::module::avg_pool1d(
                    input,
                    burn::tensor::ops::AvgPoolOptions::new([3])
                        .with_stride([1])
                        .with_padding_pairs(padding)
                        .with_dilation([2])
                        .with_count_include_pad(true)
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool1d_forward() {
        let node = create_avg_pool1d_node("pool1", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool1d_forward_with_clone() {
        let node = create_avg_pool1d_node("pool1", false);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.pool1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_avg_pool1d_field_init_ceil_mode_false() {
        let node = create_avg_pool1d_node("pool1", false);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = AvgPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_avg_pool1d_field_init_ceil_mode_true() {
        let node = create_avg_pool1d_node("pool1", true);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = AvgPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(true)
            .init();
        "#);
    }

    #[test]
    fn test_avg_pool1d_field_init_auto_pad_same_upper() {
        let config = AvgPool1dConfig::new(
            3,
            1,
            PaddingConfig1d::Valid,
            false,
            1,
            false,
            AutoPad::SameUpper,
        );
        let node = AveragePool1dNodeBuilder::new("pool1")
            .input_tensor_shape("input", vec![1, 3, 7], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = AvgPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_avg_pool1d_field_init_auto_pad_same_upper_dynamic() {
        let config = AvgPool1dConfig::new(
            3,
            1,
            PaddingConfig1d::Valid,
            false,
            1,
            false,
            AutoPad::SameUpper,
        );
        let node = AveragePool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = AvgPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Same)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_avg_pool1d_field_init_asymmetric_padding() {
        let node = create_avg_pool1d_node_asymmetric("pool1");
        let code = codegen_field_init(&node);
        // Asymmetric padding is passed directly to the module
        assert_snapshot!(code, @r"
        let pool1 = AvgPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 2))
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        ");
    }
}
