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
        let pooled = match as_float(&self.inputs[0], &input) {
            Some(float_input) => restore_dtype(
                &self.inputs[0],
                quote! { self.#field.forward(#float_input) },
            ),
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
pub(super) fn as_float(input_arg: &Argument, input: &TokenStream) -> Option<TokenStream> {
    let dtype = input_arg.ty.elem_type();
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
pub(super) fn restore_dtype(input_arg: &Argument, values: TokenStream) -> TokenStream {
    let dtype = input_arg.ty.elem_type().to_tokens();
    quote! { #values.int().cast(#dtype) }
}

/// Max pooling through burn's `max_pool1d_with_indices`, whose indices count positions
/// within one length-`L` row of the unpadded input. ONNX counts them across the whole
/// flattened input, so each row's offset is added. `storage_order` does not matter with
/// one spatial axis.
fn forward_with_indices(
    node: &onnx_ir::max_pool1d::MaxPool1dNode,
    input: TokenStream,
    output: Ident,
) -> TokenStream {
    let config = &node.config;
    let input_arg = &node.inputs[0];
    let indices_out = arg_to_ident(&node.outputs[1]);

    let padding = crate::burn::codegen::pool_padding_pairs(
        &config.auto_pad,
        &[config.padding.as_tuple()],
        &input_arg.ty,
        &input,
        &[config.kernel_size],
        &[config.stride],
        &[config.dilation],
    );
    let kernel = config.kernel_size;
    let stride = config.stride;
    let dilation = config.dilation;
    let ceil_mode = config.ceil_mode;

    // An integer input is pooled as f32, then its values are cast back.
    let (pooled_input, values) = match as_float(input_arg, &quote! { input }) {
        Some(float_input) => (float_input, restore_dtype(input_arg, quote! { values })),
        None => (quote! { input }, quote! { values }),
    };

    quote! {
        let (#output, #indices_out) = {
            // The padding may read the input's dims, so it's computed before the input
            // moves; both graph values are read before any temporary is declared.
            let (padding, input) = (#padding, #input);
            let [batch, channels, length] = input.dims();
            let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                #pooled_input,
                burn::tensor::ops::MaxPoolOptions::new([#kernel])
                    .with_stride([#stride])
                    .with_padding_pairs(padding)
                    .with_dilation([#dilation])
                    .with_ceil_mode(#ceil_mode),
            );
            let indices = indices.cast(burn::tensor::DType::I64);
            let rows = Tensor::<1, Int>::arange(
                0..(batch * channels) as i64,
                (self.device(), burn::tensor::DType::I64),
            )
            .mul_scalar(length as i64)
            .reshape([batch, channels, 1]);
            (#values, indices + rows)
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
    fn test_max_pool1d_forward_ceil_mode_asymmetric_padding() {
        let config = MaxPool1dConfig::new(
            3,
            2,
            1,
            PaddingConfig1d::Explicit(1, 2),
            true,
            AutoPad::NotSet,
        );
        let node = MaxPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
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
                let (padding, input) = ([(1usize, 1usize)], input);
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2usize])
                        .with_stride([2usize])
                        .with_padding_pairs(padding)
                        .with_dilation([1usize])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let rows = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar(length as i64)
                    .reshape([batch, channels, 1]);
                (values, indices + rows)
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
                let (padding, input) = ([(0usize, 1usize)], input);
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2usize])
                        .with_stride([2usize])
                        .with_padding_pairs(padding)
                        .with_dilation([1usize])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let rows = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar(length as i64)
                    .reshape([batch, channels, 1]);
                (values, indices + rows)
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
                let (padding, input) = ([(1usize, 1usize)], input);
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2usize])
                        .with_stride([2usize])
                        .with_padding_pairs(padding)
                        .with_dilation([1usize])
                        .with_ceil_mode(true),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let rows = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar(length as i64)
                    .reshape([batch, channels, 1]);
                (values, indices + rows)
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
                let (padding, input) = ([(0usize, 1usize)], input);
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2usize])
                        .with_stride([2usize])
                        .with_padding_pairs(padding)
                        .with_dilation([1usize])
                        .with_ceil_mode(true),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let rows = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar(length as i64)
                    .reshape([batch, channels, 1]);
                (values, indices + rows)
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
                let (padding, input) = (
                    {
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
                    },
                    input,
                );
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2usize])
                        .with_stride([1usize])
                        .with_padding_pairs(padding)
                        .with_dilation([1usize])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let rows = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar(length as i64)
                    .reshape([batch, channels, 1]);
                (values, indices + rows)
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
                let (padding, input) = ([(1usize, 1usize)], input);
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input.cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32),
                    burn::tensor::ops::MaxPoolOptions::new([2usize])
                        .with_stride([2usize])
                        .with_padding_pairs(padding)
                        .with_dilation([1usize])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let rows = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar(length as i64)
                    .reshape([batch, channels, 1]);
                (values.int().cast(burn::tensor::DType::U8), indices + rows)
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool1d_indices_same_upper_static() {
        // A 2-wide kernel on a static length resolves to (0, 1) pads at build time.
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
                let (padding, input) = ([(0usize, 1usize)], input);
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2usize])
                        .with_stride([1usize])
                        .with_padding_pairs(padding)
                        .with_dilation([1usize])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let rows = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar(length as i64)
                    .reshape([batch, channels, 1]);
                (values, indices + rows)
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
                let (padding, input) = (
                    {
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
                    },
                    input,
                );
                let [batch, channels, length] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool1d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2usize])
                        .with_stride([1usize])
                        .with_padding_pairs(padding)
                        .with_dilation([1usize])
                        .with_ceil_mode(false),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let rows = Tensor::<
                    1,
                    Int,
                >::arange(
                        0..(batch * channels) as i64,
                        (self.device(), burn::tensor::DType::I64),
                    )
                    .mul_scalar(length as i64)
                    .reshape([batch, channels, 1]);
                (values, indices + rows)
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn input_named_like_its_temporaries_is_accepted() {
        let temporaries = [
            "padding", "input", "batch", "channels", "length", "values", "indices", "rows", "dims",
            "size", "total", "small", "big",
        ];
        for temporary in temporaries {
            for (dtype, padding, auto_pad) in [
                (DType::F32, PaddingConfig1d::Explicit(1, 1), AutoPad::NotSet),
                (DType::F32, PaddingConfig1d::Explicit(0, 1), AutoPad::NotSet),
                (DType::F32, PaddingConfig1d::Valid, AutoPad::SameLower),
                (DType::U8, PaddingConfig1d::Explicit(0, 1), AutoPad::NotSet),
            ] {
                let node = MaxPool1dNodeBuilder::new("pool1")
                    .input_tensor(temporary, 3, dtype)
                    .output_tensor("output", 3, dtype)
                    .output_tensor("indices", 3, DType::I64)
                    .config(MaxPool1dConfig::new(
                        2,
                        1,
                        1,
                        padding,
                        true,
                        auto_pad.clone(),
                    ))
                    .build();
                assert_eq!(
                    shadow_check_result(&node),
                    Ok(()),
                    "input named {temporary}, {dtype:?}, {auto_pad:?}"
                );
            }
        }
    }
}
