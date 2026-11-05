use super::prelude::*;

impl NodeCodegen for onnx_ir::max_pool2d::MaxPool2dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if wants_indices(self) {
            return None;
        }
        let name = Ident::new(&self.name, Span::call_site());
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.strides.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let ceil_mode = self.config.ceil_mode;

        let input_spatial = onnx_ir::node::padding::static_spatial_dims(&self.inputs[0].ty);
        let padding = crate::burn::codegen::resolve_auto_pad_2d(
            &self.config.auto_pad,
            &self.config.padding,
            input_spatial.as_deref(),
            &self.config.kernel_size,
            &self.config.strides,
            &self.config.dilation,
        );

        Some(Field::new(
            self.name.clone(),
            quote! {
                MaxPool2d
            },
            quote! {
                let #name = MaxPool2dConfig::new(#kernel_size)
                    .with_strides(#strides)
                    .with_padding(#padding)
                    .with_dilation(#dilation)
                    .with_ceil_mode(#ceil_mode)
                    .init();
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        if wants_indices(self) {
            return forward_with_indices(self, input, output);
        }

        let field = Ident::new(&self.name, Span::call_site());
        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if wants_indices(self) {
            return;
        }
        imports.register("burn::nn::pool::MaxPool2d");
        imports.register("burn::nn::pool::MaxPool2dConfig");
        imports.register("burn::nn::PaddingConfig2d");
    }
}

/// Whether the optional ONNX Indices output is used.
fn wants_indices(node: &onnx_ir::max_pool2d::MaxPool2dNode) -> bool {
    node.outputs.get(1).is_some_and(|arg| !arg.is_optional())
}

/// Max pooling through burn's `max_pool2d_with_indices`, whose indices count positions
/// within one `H x W` plane of the unpadded input, row-major. ONNX counts them across the
/// whole flattened input, so each plane's offset is added, after transposing the in-plane
/// position for column-major `storage_order`.
fn forward_with_indices(
    node: &onnx_ir::max_pool2d::MaxPool2dNode,
    input: TokenStream,
    output: Ident,
) -> TokenStream {
    let config = &node.config;
    let indices_out = arg_to_ident(&node.outputs[1]);

    let (top, left, bottom, right) = config.padding.as_tuple();
    let padding = crate::burn::codegen::pool_padding_pairs(
        &config.auto_pad,
        &[(top, bottom), (left, right)],
        &node.inputs[0].ty,
        &input,
        &config.kernel_size,
        &config.strides,
        &config.dilation,
    );
    let kernel = config.kernel_size.to_tokens();
    let strides = config.strides.to_tokens();
    let dilation = config.dilation.to_tokens();
    let ceil_mode = config.ceil_mode;

    let in_plane = if config.storage_order == 1 {
        quote! {{
            let row = indices.clone().div_scalar(width as i64);
            let col = indices.remainder_scalar(width as i64);
            col.mul_scalar(height as i64) + row
        }}
    } else {
        quote! { indices }
    };

    quote! {
        let (#output, #indices_out) = {
            // The padding may read the input's dims, so it's computed before the input
            // moves; both graph values are read before any temporary is declared.
            let (padding, input) = (#padding, #input);
            let [batch, channels, height, width] = input.dims();
            let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                input,
                burn::tensor::ops::MaxPoolOptions::new(#kernel)
                    .with_stride(#strides)
                    .with_padding_pairs(padding)
                    .with_dilation(#dilation)
                    .with_ceil_mode(#ceil_mode),
            );
            let indices = indices.cast(burn::tensor::DType::I64);
            let planes = Tensor::<1, Int>::arange(
                0..(batch * channels) as i64,
                (self.device(), burn::tensor::DType::I64),
            )
            .mul_scalar((height * width) as i64)
            .reshape([batch, channels, 1, 1]);
            (values, #in_plane + planes)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::max_pool2d::{MaxPool2dConfig, MaxPool2dNode, MaxPool2dNodeBuilder};
    use onnx_ir::padding::{AutoPad, PaddingConfig2d};

    fn create_max_pool2d_node(name: &str, ceil_mode: bool) -> MaxPool2dNode {
        let config = MaxPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Valid,
            [1, 1],
            ceil_mode,
            AutoPad::NotSet,
        );

        MaxPool2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    fn create_max_pool2d_node_asymmetric(name: &str) -> MaxPool2dNode {
        // Asymmetric padding: top=1, left=2, bottom=3, right=4
        let config = MaxPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Explicit(1, 2, 3, 4),
            [1, 1],
            false,
            AutoPad::NotSet,
        );

        MaxPool2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_max_pool2d_forward() {
        let node = create_max_pool2d_node("pool1", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool2d_forward_with_clone() {
        let node = create_max_pool2d_node("pool1", false);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.pool1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_max_pool2d_field_init_ceil_mode_false() {
        let node = create_max_pool2d_node("pool1", false);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool2d_field_init_ceil_mode_true() {
        let node = create_max_pool2d_node("pool1", true);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(true)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool2d_forward_asymmetric_padding() {
        let node = create_max_pool2d_node_asymmetric("pool1");
        let code = codegen_forward_default(&node);
        // Asymmetric padding is now handled by the burn-nn module
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool2d_forward_ceil_mode_asymmetric_padding() {
        let config = MaxPool2dConfig::new(
            [3, 3],
            [2, 2],
            PaddingConfig2d::Explicit(1, 2, 3, 4),
            [1, 1],
            true,
            AutoPad::NotSet,
        );
        let node = MaxPool2dNodeBuilder::new("pool1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool2d_field_init_auto_pad_same_upper() {
        let config = MaxPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Valid,
            [1, 1],
            false,
            AutoPad::SameUpper,
        );
        let node = MaxPool2dNodeBuilder::new("pool1")
            .input_tensor_shape("input", vec![1, 3, 7, 7], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool2d_field_init_auto_pad_same_upper_dynamic() {
        let config = MaxPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Valid,
            [1, 1],
            false,
            AutoPad::SameUpper,
        );
        let node = MaxPool2dNodeBuilder::new("pool1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool2d_field_init_asymmetric_padding() {
        let node = create_max_pool2d_node_asymmetric("pool1");
        let code = codegen_field_init(&node);
        // Asymmetric padding is passed directly to the module
        assert_snapshot!(code, @r"
        let pool1 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 2, 3, 4))
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        ");
    }

    fn create_max_pool2d_indices_node(storage_order: i64) -> MaxPool2dNode {
        let mut config = MaxPool2dConfig::new(
            [2, 2],
            [2, 2],
            PaddingConfig2d::Explicit(1, 1, 1, 1),
            [1, 1],
            false,
            AutoPad::NotSet,
        );
        config.storage_order = storage_order;

        MaxPool2dNodeBuilder::new("pool1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("indices", 4, DType::I64)
            .config(config)
            .build()
    }

    #[test]
    fn test_max_pool2d_indices() {
        let code = codegen_forward_default(&create_max_pool2d_indices_node(0));
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> (Tensor<4>, Tensor<4, Int>) {
            let (output, indices) = {
                let (padding, input) = ([(1usize, 1usize), (1usize, 1usize)], input);
                let [batch, channels, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2, 2])
                        .with_stride([2, 2])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let planes = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar((height * width) as i64)
                    .reshape([batch, channels, 1, 1]);
                (values, indices + planes)
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool2d_indices_column_major() {
        let code = codegen_forward_default(&create_max_pool2d_indices_node(1));
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> (Tensor<4>, Tensor<4, Int>) {
            let (output, indices) = {
                let (padding, input) = ([(1usize, 1usize), (1usize, 1usize)], input);
                let [batch, channels, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2, 2])
                        .with_stride([2, 2])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let planes = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar((height * width) as i64)
                    .reshape([batch, channels, 1, 1]);
                (
                    values,
                    {
                        let row = indices.clone().div_scalar(width as i64);
                        let col = indices.remainder_scalar(width as i64);
                        col.mul_scalar(height as i64) + row
                    } + planes,
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool2d_indices_asymmetric_padding() {
        let config = MaxPool2dConfig::new(
            [2, 2],
            [1, 1],
            PaddingConfig2d::Explicit(0, 0, 1, 1),
            [1, 1],
            false,
            AutoPad::NotSet,
        );
        let node = MaxPool2dNodeBuilder::new("pool1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("indices", 4, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> (Tensor<4>, Tensor<4, Int>) {
            let (output, indices) = {
                let (padding, input) = ([(0usize, 1usize), (0usize, 1usize)], input);
                let [batch, channels, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2, 2])
                        .with_stride([1, 1])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let planes = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar((height * width) as i64)
                    .reshape([batch, channels, 1, 1]);
                (values, indices + planes)
            };
            (output, indices)
        }
        ");
    }

    fn ceil_indices_node(padding: PaddingConfig2d) -> MaxPool2dNode {
        let config = MaxPool2dConfig::new([2, 2], [2, 2], padding, [1, 1], true, AutoPad::NotSet);
        MaxPool2dNodeBuilder::new("pool1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("indices", 4, DType::I64)
            .config(config)
            .build()
    }

    #[test]
    fn test_max_pool2d_indices_ceil_mode_symmetric() {
        let node = ceil_indices_node(PaddingConfig2d::Explicit(1, 1, 1, 1));
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<4>) -> (Tensor<4>, Tensor<4, Int>) {
            let (output, indices) = {
                let (padding, input) = ([(1usize, 1usize), (1usize, 1usize)], input);
                let [batch, channels, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2, 2])
                        .with_stride([2, 2])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_ceil_mode(true),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let planes = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar((height * width) as i64)
                    .reshape([batch, channels, 1, 1]);
                (values, indices + planes)
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool2d_indices_ceil_mode_asymmetric() {
        let node = ceil_indices_node(PaddingConfig2d::Explicit(0, 0, 1, 1));
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<4>) -> (Tensor<4>, Tensor<4, Int>) {
            let (output, indices) = {
                let (padding, input) = ([(0usize, 1usize), (0usize, 1usize)], input);
                let [batch, channels, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2, 2])
                        .with_stride([2, 2])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 1])
                        .with_ceil_mode(true),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let planes = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar((height * width) as i64)
                    .reshape([batch, channels, 1, 1]);
                (values, indices + planes)
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn input_named_like_its_temporaries_is_accepted() {
        let temporaries = [
            "padding", "input", "batch", "channels", "height", "width", "values", "indices",
            "planes", "row", "col", "dims", "size", "total", "small", "big",
        ];
        for temporary in temporaries {
            for (padding, auto_pad, storage_order) in [
                (PaddingConfig2d::Explicit(1, 1, 1, 1), AutoPad::NotSet, 0),
                (PaddingConfig2d::Explicit(0, 0, 1, 1), AutoPad::NotSet, 1),
                (PaddingConfig2d::Valid, AutoPad::SameUpper, 1),
            ] {
                let mut config =
                    MaxPool2dConfig::new([2, 2], [1, 1], padding, [1, 1], true, auto_pad.clone());
                config.storage_order = storage_order;
                let node = MaxPool2dNodeBuilder::new("pool1")
                    .input_tensor(temporary, 4, DType::F32)
                    .output_tensor("output", 4, DType::F32)
                    .output_tensor("indices", 4, DType::I64)
                    .config(config)
                    .build();
                assert_eq!(
                    shadow_check_result(&node),
                    Ok(()),
                    "input named {temporary}, {auto_pad:?}, storage_order {storage_order}"
                );
            }
        }
    }
}
