use super::prelude::*;

impl NodeCodegen for onnx_ir::max_pool1d::MaxPool1dNode {
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
        let strides = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
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
                MaxPool1d
            },
            quote! {
                let #name = MaxPool1dConfig::new(#kernel_size)
                    .with_stride(#strides)
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
        let pooled = match as_float(self, &input) {
            Some(float_input) => restore_dtype(self, quote! { self.#field.forward(#float_input) }),
            None => quote! { self.#field.forward(#input) },
        };
        quote! {
            let #output = #pooled;
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if wants_indices(self) {
            return;
        }
        imports.register("burn::nn::pool::MaxPool1d");
        imports.register("burn::nn::pool::MaxPool1dConfig");
        imports.register("burn::nn::PaddingConfig1d");
    }
}

/// Whether the optional ONNX Indices output is present (not omitted).
fn wants_indices(node: &onnx_ir::max_pool1d::MaxPool1dNode) -> bool {
    node.outputs.get(1).is_some_and(|arg| !arg.is_optional())
}

/// burn pools float tensors only. An integer input (int8/uint8 since opset 12) is
/// pooled as f32, which holds it exactly; returns that conversion, or `None` for a
/// float input.
fn as_float(node: &onnx_ir::max_pool1d::MaxPool1dNode, input: &TokenStream) -> Option<TokenStream> {
    let dtype = node.inputs[0].ty.elem_type();
    if dtype.is_float() {
        None
    } else if dtype.is_uint() {
        Some(
            quote! { #input.cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32) },
        )
    } else {
        Some(quote! { #input.float().cast(burn::tensor::DType::F32) })
    }
}

/// Casts pooled f32 `values` back to the integer input's dtype (see [`as_float`]).
fn restore_dtype(node: &onnx_ir::max_pool1d::MaxPool1dNode, values: TokenStream) -> TokenStream {
    let dtype = node.inputs[0].ty.elem_type().to_tokens();
    quote! { #values.int().cast(#dtype) }
}

/// Max pooling through burn's `max_pool1d_with_indices`, whose indices count positions
/// within one length-`L` row. ONNX counts them across the whole flattened input, so
/// each row's offset is added. `storage_order` does not matter with one spatial axis.
///
/// burn pads both ends equally. Other padding (asymmetric, or SAME on an input sized
/// only at run time) is applied beforehand with -inf, which never beats a finite value,
/// and the indices are shifted back from the padded row to the input's.
fn forward_with_indices(
    node: &onnx_ir::max_pool1d::MaxPool1dNode,
    input: TokenStream,
    output: Ident,
) -> TokenStream {
    let config = &node.config;
    let indices_out = arg_to_ident(&node.outputs[1]);

    // An integer input is converted once up front and read from `float_input`.
    let (convert, input, values) = match as_float(node, &input) {
        Some(float_input) => (
            quote! { let float_input = #float_input; },
            quote! { float_input },
            restore_dtype(node, quote! { values }),
        ),
        None => (quote! {}, input, quote! { values }),
    };

    let input_spatial = onnx_ir::node::padding::static_spatial_dims(&node.inputs[0].ty);
    let padding = crate::burn::codegen::resolve_padding_pairs(
        &config.auto_pad,
        &[config.padding.as_tuple()],
        input_spatial.as_deref(),
        &[config.kernel_size],
        &[config.stride],
        &[config.dilation],
    );

    let kernel = config.kernel_size;
    let stride = config.stride;
    let dilation = config.dilation;
    let ceil_mode = config.ceil_mode;
    let rows = quote! {
        Tensor::<1, Int>::arange(
            0..(batch * channels) as i64,
            (&self.device, burn::tensor::DType::I64),
        )
        .mul_scalar(length as i64)
        .reshape([batch, channels, 1])
    };

    // ONNX drops a ceil-mode window that would start inside the trailing padding.
    // burn keeps it when it pads (tracel-ai/burn#5791), and when the input is
    // pre-padded burn cannot tell the padding from input, so both outputs are cut
    // back to the ONNX size.
    // Reads `length`, `left`, `right`; rebinds `values` and `indices`.
    let trim = config.ceil_mode.then(|| {
        quote! {
            let out_len = {
                let len = (length + left + right - (#kernel - 1) * #dilation - 1).div_ceil(#stride) + 1;
                if (len - 1) * #stride >= length + left { len - 1 } else { len }
            };
            let values = values.slice(s![.., .., 0..out_len]);
            let indices = indices.slice(s![.., .., 0..out_len]);
        }
    });

    // Symmetric padding known at build time: burn pads, and its indices already
    // address the input row.
    if let Some(&[(l, r)]) = padding.as_deref()
        && l == r
    {
        let sym_trim = trim.map(|trim| {
            quote! {
                let (left, right) = (#l, #l);
                #trim
            }
        });
        return quote! {
            let (#output, #indices_out) = {
                #convert
                let [batch, channels, length] = #input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    #input,
                    #kernel,
                    #stride,
                    #l,
                    #dilation,
                    #ceil_mode,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                #sym_trim
                (#values, indices + #rows)
            };
        };
    }

    let padding = match padding {
        Some(pairs) => crate::burn::codegen::padding_pairs_tokens(&pairs),
        None => crate::burn::codegen::runtime_same_padding(
            &config.auto_pad,
            &input,
            &[config.kernel_size],
            &[config.stride],
            &[config.dilation],
        ),
    };

    quote! {
        let (#output, #indices_out) = {
            #convert
            let [batch, channels, length] = #input.dims();
            let [(left, right)] = #padding;
            let padded = #input.pad(
                [(0, 0), (0, 0), (left, right)],
                burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
            );
            let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                padded,
                #kernel,
                #stride,
                0,
                #dilation,
                #ceil_mode,
            );
            let indices = indices.cast(burn::tensor::DType::I64);
            #trim
            (#values, indices.sub_scalar(left as i64) + #rows)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::max_pool1d::{MaxPool1dConfig, MaxPool1dNode, MaxPool1dNodeBuilder};
    use onnx_ir::padding::{AutoPad, PaddingConfig1d};

    fn create_max_pool1d_node(name: &str, ceil_mode: bool) -> MaxPool1dNode {
        let config =
            MaxPool1dConfig::new(3, 1, 1, PaddingConfig1d::Valid, ceil_mode, AutoPad::NotSet);

        MaxPool1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    fn create_max_pool1d_node_asymmetric(name: &str) -> MaxPool1dNode {
        // Asymmetric padding: left=1, right=2
        let config = MaxPool1dConfig::new(
            3,
            1,
            1,
            PaddingConfig1d::Explicit(1, 2),
            false,
            AutoPad::NotSet,
        );

        MaxPool1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_max_pool1d_forward() {
        let node = create_max_pool1d_node("pool1", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool1d_forward_with_clone() {
        let node = create_max_pool1d_node("pool1", false);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.pool1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_max_pool1d_field_init_ceil_mode_false() {
        let node = create_max_pool1d_node("pool1", false);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool1d_field_init_ceil_mode_true() {
        let node = create_max_pool1d_node("pool1", true);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_ceil_mode(true)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool1d_forward_asymmetric_padding() {
        let node = create_max_pool1d_node_asymmetric("pool1");
        let code = codegen_forward_default(&node);
        // Asymmetric padding is now handled by the burn-nn module
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool1d_field_init_auto_pad_same_upper() {
        let config =
            MaxPool1dConfig::new(3, 1, 1, PaddingConfig1d::Valid, false, AutoPad::SameUpper);
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor_shape("input", vec![1, 3, 7], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool1d_field_init_auto_pad_same_upper_dynamic() {
        let config =
            MaxPool1dConfig::new(3, 1, 1, PaddingConfig1d::Valid, false, AutoPad::SameUpper);
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Same)
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool1d_field_init_asymmetric_padding() {
        let node = create_max_pool1d_node_asymmetric("pool1");
        let code = codegen_field_init(&node);
        // Asymmetric padding is passed directly to the module
        assert_snapshot!(code, @r"
        let pool1 = MaxPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 2))
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        ");
    }

    fn indices_node(padding: PaddingConfig1d, ceil_mode: bool) -> MaxPool1dNode {
        let config = MaxPool1dConfig::new(2, 2, 1, padding, ceil_mode, AutoPad::NotSet);
        MaxPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .output_tensor("indices", 3, DType::I64)
            .config(config)
            .build()
    }

    #[test]
    fn test_max_pool1d_indices() {
        let node = indices_node(PaddingConfig1d::Explicit(1, 1), false);
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<3>, Tensor<3, Int>) {
            let (output, indices) = {
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    2usize,
                    2usize,
                    1usize,
                    1usize,
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
                            .mul_scalar(length as i64)
                            .reshape([batch, channels, 1]),
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool1d_indices_asymmetric_padding() {
        let node = indices_node(PaddingConfig1d::Explicit(0, 1), false);
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<3>, Tensor<3, Int>) {
            let (output, indices) = {
                let [batch, channels, length] = input.dims();
                let [(left, right)] = [(0usize, 1usize)];
                let padded = input
                    .pad(
                        [(0, 0), (0, 0), (left, right)],
                        burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
                    );
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    padded,
                    2usize,
                    2usize,
                    0,
                    1usize,
                    false,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                (
                    values,
                    indices.sub_scalar(left as i64)
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar(length as i64)
                            .reshape([batch, channels, 1]),
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool1d_indices_ceil_mode_symmetric() {
        let node = indices_node(PaddingConfig1d::Explicit(1, 1), true);
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<3>, Tensor<3, Int>) {
            let (output, indices) = {
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    2usize,
                    2usize,
                    1usize,
                    1usize,
                    true,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let (left, right) = (1usize, 1usize);
                let out_len = {
                    let len = (length + left + right - (2usize - 1) * 1usize - 1)
                        .div_ceil(2usize) + 1;
                    if (len - 1) * 2usize >= length + left { len - 1 } else { len }
                };
                let values = values.slice(s![.., .., 0..out_len]);
                let indices = indices.slice(s![.., .., 0..out_len]);
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
                            .mul_scalar(length as i64)
                            .reshape([batch, channels, 1]),
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool1d_indices_ceil_mode_asymmetric() {
        let node = indices_node(PaddingConfig1d::Explicit(0, 1), true);
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<3>, Tensor<3, Int>) {
            let (output, indices) = {
                let [batch, channels, length] = input.dims();
                let [(left, right)] = [(0usize, 1usize)];
                let padded = input
                    .pad(
                        [(0, 0), (0, 0), (left, right)],
                        burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
                    );
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    padded,
                    2usize,
                    2usize,
                    0,
                    1usize,
                    true,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let out_len = {
                    let len = (length + left + right - (2usize - 1) * 1usize - 1)
                        .div_ceil(2usize) + 1;
                    if (len - 1) * 2usize >= length + left { len - 1 } else { len }
                };
                let values = values.slice(s![.., .., 0..out_len]);
                let indices = indices.slice(s![.., .., 0..out_len]);
                (
                    values,
                    indices.sub_scalar(left as i64)
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar(length as i64)
                            .reshape([batch, channels, 1]),
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool1d_indices_same_upper_dynamic() {
        let config =
            MaxPool1dConfig::new(2, 1, 1, PaddingConfig1d::Valid, false, AutoPad::SameUpper);
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .output_tensor("indices", 3, DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<3>, Tensor<3, Int>) {
            let (output, indices) = {
                let [batch, channels, length] = input.dims();
                let [(left, right)] = {
                    let dims = input.dims();
                    [
                        {
                            let size = dims[2usize];
                            let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                + 2usize)
                                .saturating_sub(size);
                            let small = total / 2;
                            let big = total - small;
                            (small, big)
                        },
                    ]
                };
                let padded = input
                    .pad(
                        [(0, 0), (0, 0), (left, right)],
                        burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
                    );
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    padded,
                    2usize,
                    1usize,
                    0,
                    1usize,
                    false,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                (
                    values,
                    indices.sub_scalar(left as i64)
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar(length as i64)
                            .reshape([batch, channels, 1]),
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool1d_omitted_indices_uses_module() {
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .output_tensor("", 3, DType::I64)
            .config(MaxPool1dConfig::new(
                2,
                2,
                1,
                PaddingConfig1d::Valid,
                false,
                AutoPad::NotSet,
            ))
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool1d_forward_int8() {
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::I8)
            .output_tensor("output", 3, DType::I8)
            .config(MaxPool1dConfig::new(
                3,
                1,
                1,
                PaddingConfig1d::Valid,
                false,
                AutoPad::NotSet,
            ))
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3, Int>) -> Tensor<3, Int> {
            let output = self
                .pool1
                .forward(input.float().cast(burn::tensor::DType::F32))
                .int()
                .cast(burn::tensor::DType::I8);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool1d_indices_uint8() {
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::U8)
            .output_tensor("output", 3, DType::U8)
            .output_tensor("indices", 3, DType::I64)
            .config(MaxPool1dConfig::new(
                2,
                2,
                1,
                PaddingConfig1d::Explicit(1, 1),
                false,
                AutoPad::NotSet,
            ))
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3, Int>) -> (Tensor<3, Int>, Tensor<3, Int>) {
            let (output, indices) = {
                let float_input = input
                    .cast(burn::tensor::DType::I32)
                    .float()
                    .cast(burn::tensor::DType::F32);
                let [batch, channels, length] = float_input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    float_input,
                    2usize,
                    2usize,
                    1usize,
                    1usize,
                    false,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                (
                    values.int().cast(burn::tensor::DType::U8),
                    indices
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar(length as i64)
                            .reshape([batch, channels, 1]),
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool1d_indices_same_upper_static() {
        // A 2-wide kernel on a static length pads (0, 1), so the input is pre-padded.
        let config =
            MaxPool1dConfig::new(2, 1, 1, PaddingConfig1d::Valid, false, AutoPad::SameUpper);
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor_shape("input", vec![1, 3, 6], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .output_tensor("indices", 3, DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<3>, Tensor<3, Int>) {
            let (output, indices) = {
                let [batch, channels, length] = input.dims();
                let [(left, right)] = [(0usize, 1usize)];
                let padded = input
                    .pad(
                        [(0, 0), (0, 0), (left, right)],
                        burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
                    );
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    padded,
                    2usize,
                    1usize,
                    0,
                    1usize,
                    false,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                (
                    values,
                    indices.sub_scalar(left as i64)
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar(length as i64)
                            .reshape([batch, channels, 1]),
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool1d_indices_same_lower_dynamic() {
        let config =
            MaxPool1dConfig::new(2, 1, 1, PaddingConfig1d::Valid, false, AutoPad::SameLower);
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .output_tensor("indices", 3, DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<3>, Tensor<3, Int>) {
            let (output, indices) = {
                let [batch, channels, length] = input.dims();
                let [(left, right)] = {
                    let dims = input.dims();
                    [
                        {
                            let size = dims[2usize];
                            let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                + 2usize)
                                .saturating_sub(size);
                            let small = total / 2;
                            let big = total - small;
                            (big, small)
                        },
                    ]
                };
                let padded = input
                    .pad(
                        [(0, 0), (0, 0), (left, right)],
                        burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
                    );
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    padded,
                    2usize,
                    1usize,
                    0,
                    1usize,
                    false,
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                (
                    values,
                    indices.sub_scalar(left as i64)
                        + Tensor::<
                            1,
                            Int,
                        >::arange(
                                0..(batch * channels) as i64,
                                (&self.device, burn::tensor::DType::I64),
                            )
                            .mul_scalar(length as i64)
                            .reshape([batch, channels, 1]),
                )
            };
            (output, indices)
        }
        ");
    }
}
