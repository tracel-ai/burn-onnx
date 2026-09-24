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
/// within one `H x W` plane. ONNX counts them across the whole flattened input, so
/// each plane's offset is added, after transposing the in-plane position for
/// column-major `storage_order`.
///
/// burn pads both sides of an axis equally. Other padding (asymmetric, or SAME on an
/// input sized only at run time) is applied beforehand with -inf, which never wins
/// the max, and the indices are mapped back from the padded plane to the input's.
fn forward_with_indices(
    node: &onnx_ir::max_pool2d::MaxPool2dNode,
    input: TokenStream,
    output: Ident,
) -> TokenStream {
    let config = &node.config;
    let indices_out = arg_to_ident(&node.outputs[1]);

    let (top, left, bottom, right) = config.padding.as_tuple();
    let input_spatial = onnx_ir::node::padding::static_spatial_dims(&node.inputs[0].ty);
    let padding = crate::burn::codegen::resolve_padding_pairs(
        &config.auto_pad,
        &[(top, bottom), (left, right)],
        input_spatial.as_deref(),
        &config.kernel_size,
        &config.strides,
        &config.dilation,
    );

    let kernel = config.kernel_size.to_tokens();
    let strides = config.strides.to_tokens();
    let dilation = config.dilation.to_tokens();
    let ceil_mode = config.ceil_mode;
    let in_plane = if config.storage_order == 1 {
        quote! { col.mul_scalar(height as i64) + row }
    } else {
        quote! { row.mul_scalar(width as i64) + col }
    };
    let planes = quote! {
        Tensor::<1, Int>::arange(
            0..(batch * channels) as i64,
            (&self.device, burn::tensor::DType::I64),
        )
        .mul_scalar((height * width) as i64)
        .reshape([batch, channels, 1, 1])
    };

    // ONNX drops a ceil-mode window that would start inside the trailing padding.
    // burn keeps it when it pads (tracel-ai/burn#5791), and when the input is
    // pre-padded burn cannot tell the padding from input, so both outputs are cut
    // back to the ONNX size.
    // Reads `height`, `width` and the pads bound as `top`, `bottom`, `left`, `right`.
    let trim = config.ceil_mode.then(|| {
        let [kh, kw] = config.kernel_size;
        let [sh, sw] = config.strides;
        let [dh, dw] = config.dilation;
        quote! {
            let out_len = |size: usize, begin: usize, end: usize, kernel: usize, stride: usize, dilation: usize| {
                let len = (size + begin + end - (kernel - 1) * dilation - 1).div_ceil(stride) + 1;
                if (len - 1) * stride >= size + begin { len - 1 } else { len }
            };
            let out_h = out_len(height, top, bottom, #kh, #sh, #dh);
            let out_w = out_len(width, left, right, #kw, #sw, #dw);
            let values = values.slice(s![.., .., 0..out_h, 0..out_w]);
            let indices = indices.slice(s![.., .., 0..out_h, 0..out_w]);
        }
    });

    // Symmetric padding known at build time: burn pads, and its indices already
    // address the input plane in row-major order.
    if let Some(&[(t, b), (l, r)]) = padding.as_deref()
        && t == b
        && l == r
    {
        let in_plane = if config.storage_order == 1 {
            quote! {{
                let row = indices.clone().div_scalar(width as i64);
                let col = indices.remainder_scalar(width as i64);
                #in_plane
            }}
        } else {
            quote! { indices }
        };
        let sym_trim = trim.map(|trim| {
            quote! {
                let (top, bottom, left, right) = (#t, #t, #l, #l);
                #trim
            }
        });
        return quote! {
            let (#output, #indices_out) = {
                let [batch, channels, height, width] = #input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    #input,
                    #kernel,
                    #strides,
                    [#t, #l],
                    #dilation,
                    #ceil_mode,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                #sym_trim
                (values, #in_plane + #planes)
            };
        };
    }

    let padding = match padding {
        Some(pairs) => crate::burn::codegen::padding_pairs_tokens(&pairs),
        None => crate::burn::codegen::runtime_same_padding(
            &config.auto_pad,
            &input,
            &config.kernel_size,
            &config.strides,
            &config.dilation,
        ),
    };

    quote! {
        let (#output, #indices_out) = {
            let [batch, channels, height, width] = #input.dims();
            let [(top, bottom), (left, right)] = #padding;
            let padded = #input.pad(
                [(0, 0), (0, 0), (top, bottom), (left, right)],
                burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
            );
            let padded_width = width + left + right;
            let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                padded,
                #kernel,
                #strides,
                [0, 0],
                #dilation,
                #ceil_mode,
            );
            let indices = indices.cast(burn::tensor::DType::I64);
            #trim
            let row = indices.clone().div_scalar(padded_width as i64).sub_scalar(top as i64);
            let col = indices
                .remainder_scalar(padded_width as i64)
                .sub_scalar(left as i64);
            (values, #in_plane + #planes)
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
                let [batch, channels, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    input,
                    [2, 2],
                    [2, 2],
                    [1usize, 1usize],
                    [1, 1],
                    false,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                (
                    values,
                    indices
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar((height * width) as i64)
                            .reshape([batch, channels, 1, 1]),
                )
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
                let [batch, channels, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    input,
                    [2, 2],
                    [2, 2],
                    [1usize, 1usize],
                    [1, 1],
                    false,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                (
                    values,
                    {
                        let row = indices.clone().div_scalar(width as i64);
                        let col = indices.remainder_scalar(width as i64);
                        col.mul_scalar(height as i64) + row
                    }
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar((height * width) as i64)
                            .reshape([batch, channels, 1, 1]),
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
                let [batch, channels, height, width] = input.dims();
                let [(top, bottom), (left, right)] = [(0usize, 1usize), (0usize, 1usize)];
                let padded = input
                    .pad(
                        [(0, 0), (0, 0), (top, bottom), (left, right)],
                        burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
                    );
                let padded_width = width + left + right;
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    padded,
                    [2, 2],
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    false,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let row = indices.clone().div_scalar(padded_width as i64).sub_scalar(top as i64);
                let col = indices.remainder_scalar(padded_width as i64).sub_scalar(left as i64);
                (
                    values,
                    row.mul_scalar(width as i64) + col
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar((height * width) as i64)
                            .reshape([batch, channels, 1, 1]),
                )
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
                let [batch, channels, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    input,
                    [2, 2],
                    [2, 2],
                    [1usize, 1usize],
                    [1, 1],
                    true,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let (top, bottom, left, right) = (1usize, 1usize, 1usize, 1usize);
                let out_len = |
                    size: usize,
                    begin: usize,
                    end: usize,
                    kernel: usize,
                    stride: usize,
                    dilation: usize|
                {
                    let len = (size + begin + end - (kernel - 1) * dilation - 1).div_ceil(stride)
                        + 1;
                    if (len - 1) * stride >= size + begin { len - 1 } else { len }
                };
                let out_h = out_len(height, top, bottom, 2usize, 2usize, 1usize);
                let out_w = out_len(width, left, right, 2usize, 2usize, 1usize);
                let values = values.slice(s![.., .., 0..out_h, 0..out_w]);
                let indices = indices.slice(s![.., .., 0..out_h, 0..out_w]);
                (
                    values,
                    indices
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar((height * width) as i64)
                            .reshape([batch, channels, 1, 1]),
                )
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
                let [batch, channels, height, width] = input.dims();
                let [(top, bottom), (left, right)] = [(0usize, 1usize), (0usize, 1usize)];
                let padded = input
                    .pad(
                        [(0, 0), (0, 0), (top, bottom), (left, right)],
                        burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
                    );
                let padded_width = width + left + right;
                let (values, indices) = burn::tensor::module::max_pool2d_with_indices(
                    padded,
                    [2, 2],
                    [2, 2],
                    [0, 0],
                    [1, 1],
                    true,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let out_len = |
                    size: usize,
                    begin: usize,
                    end: usize,
                    kernel: usize,
                    stride: usize,
                    dilation: usize|
                {
                    let len = (size + begin + end - (kernel - 1) * dilation - 1).div_ceil(stride)
                        + 1;
                    if (len - 1) * stride >= size + begin { len - 1 } else { len }
                };
                let out_h = out_len(height, top, bottom, 2usize, 2usize, 1usize);
                let out_w = out_len(width, left, right, 2usize, 2usize, 1usize);
                let values = values.slice(s![.., .., 0..out_h, 0..out_w]);
                let indices = indices.slice(s![.., .., 0..out_h, 0..out_w]);
                let row = indices.clone().div_scalar(padded_width as i64).sub_scalar(top as i64);
                let col = indices.remainder_scalar(padded_width as i64).sub_scalar(left as i64);
                (
                    values,
                    row.mul_scalar(width as i64) + col
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar((height * width) as i64)
                            .reshape([batch, channels, 1, 1]),
                )
            };
            (output, indices)
        }
        ");
    }
}
