use super::avg_pool3d::pool3d_padding;
use super::max_pool1d::{as_float, restore_dtype};
use super::prelude::*;

impl NodeCodegen for onnx_ir::node::max_pool3d::MaxPool3dNode {
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
        let dilation = config.dilation.to_tokens();
        let ceil_mode = config.ceil_mode;
        let options = quote! {
            burn::tensor::ops::MaxPoolOptions::new(#kernel)
                .with_stride(#strides)
                .with_padding_pairs(padding)
                .with_dilation(#dilation)
                .with_ceil_mode(#ceil_mode)
        };

        // burn pools float tensors only: an integer input is pooled as f32, then cast back.
        let float_input = as_float(input_arg, &quote! { input });
        let restore = |values: TokenStream| match float_input {
            Some(_) => restore_dtype(input_arg, values),
            None => values,
        };
        let pooled_input = float_input.clone().unwrap_or(quote! { input });

        if !wants_indices(self) {
            let pooled = restore(quote! {
                burn::tensor::module::max_pool3d(#pooled_input, #options)
            });
            return quote! {
                let #output = {
                    // The padding may read the input's dims, so it's computed before the input
                    // moves; both graph values are read before any temporary is declared.
                    let (padding, input) = (#padding, #input);
                    #pooled
                };
            };
        }

        // burn's indices count positions within one D x H x W volume, row-major. ONNX counts
        // them across the whole flattened input, so each volume's offset is added, after
        // reordering the in-volume position to column-major for `storage_order = 1`.
        let indices_out = arg_to_ident(&self.outputs[1]);
        let in_volume = if config.storage_order == 1 {
            quote! {{
                let d = indices.clone().div_scalar((height * width) as i64);
                let h = indices.clone().div_scalar(width as i64).remainder_scalar(height as i64);
                let w = indices.remainder_scalar(width as i64);
                w.mul_scalar((depth * height) as i64) + h.mul_scalar(depth as i64) + d
            }}
        } else {
            quote! { indices }
        };
        let values = restore(quote! { values });
        quote! {
            let (#output, #indices_out) = {
                let (padding, input) = (#padding, #input);
                let [batch, channels, depth, height, width] = input.dims();
                let (values, indices) =
                    burn::tensor::module::max_pool3d_with_indices(#pooled_input, #options);
                let indices = indices.cast(burn::tensor::DType::I64);
                let volumes = Tensor::<1, Int>::arange(
                    0..(batch * channels) as i64,
                    (&self.device, burn::tensor::DType::I64),
                )
                .mul_scalar((depth * height * width) as i64)
                .reshape([batch, channels, 1, 1, 1]);
                (#values, #in_volume + volumes)
            };
        }
    }
}

/// Whether the optional ONNX Indices output is used.
fn wants_indices(node: &onnx_ir::node::max_pool3d::MaxPool3dNode) -> bool {
    node.outputs.get(1).is_some_and(|arg| !arg.is_optional())
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::max_pool3d::{MaxPool3dConfig, MaxPool3dNode, MaxPool3dNodeBuilder};
    use onnx_ir::node::padding::{AutoPad, PaddingConfig3d};

    fn node(input: &str, indices: bool, storage_order: i64) -> MaxPool3dNode {
        typed_node(input, DType::F32, indices, storage_order)
    }

    fn typed_node(input: &str, dtype: DType, indices: bool, storage_order: i64) -> MaxPool3dNode {
        let mut config = MaxPool3dConfig::new(
            [2, 3, 3],
            [2, 2, 2],
            PaddingConfig3d::Explicit(1, 0, 1, 0, 1, 1),
            [1, 2, 1],
            true,
            AutoPad::NotSet,
        );
        config.storage_order = storage_order;
        let builder = MaxPool3dNodeBuilder::new("maxpool1")
            .input_tensor(input, 5, dtype)
            .output_tensor("output", 5, dtype);
        let builder = if indices {
            builder.output_tensor("indices", 5, DType::I64)
        } else {
            builder
        };
        builder.config(config).build()
    }

    #[test]
    fn test_max_pool3d() {
        let code = codegen_forward_default(&node("input", false, 0));
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = {
                let (padding, input) = (
                    [(1usize, 0usize), (0usize, 1usize), (1usize, 1usize)],
                    input,
                );
                burn::tensor::module::max_pool3d(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2, 3, 3])
                        .with_stride([2, 2, 2])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 2, 1])
                        .with_ceil_mode(true),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_max_pool3d_indices() {
        let code = codegen_forward_default(&node("input", true, 0));
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> (Tensor<5>, Tensor<5, Int>) {
            let (output, indices) = {
                let (padding, input) = (
                    [(1usize, 0usize), (0usize, 1usize), (1usize, 1usize)],
                    input,
                );
                let [batch, channels, depth, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool3d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2, 3, 3])
                        .with_stride([2, 2, 2])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 2, 1])
                        .with_ceil_mode(true),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let volumes = Tensor::<
                    1,
                    Int,
                >::arange(0..(batch * channels) as i64, (&self.device, burn::tensor::DType::I64))
                    .mul_scalar((depth * height * width) as i64)
                    .reshape([batch, channels, 1, 1, 1]);
                (values, indices + volumes)
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool3d_indices_column_major() {
        let code = codegen_forward_default(&node("input", true, 1));
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> (Tensor<5>, Tensor<5, Int>) {
            let (output, indices) = {
                let (padding, input) = (
                    [(1usize, 0usize), (0usize, 1usize), (1usize, 1usize)],
                    input,
                );
                let [batch, channels, depth, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool3d_with_indices(
                    input,
                    burn::tensor::ops::MaxPoolOptions::new([2, 3, 3])
                        .with_stride([2, 2, 2])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 2, 1])
                        .with_ceil_mode(true),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let volumes = Tensor::<
                    1,
                    Int,
                >::arange(0..(batch * channels) as i64, (&self.device, burn::tensor::DType::I64))
                    .mul_scalar((depth * height * width) as i64)
                    .reshape([batch, channels, 1, 1, 1]);
                (
                    values,
                    {
                        let d = indices.clone().div_scalar((height * width) as i64);
                        let h = indices
                            .clone()
                            .div_scalar(width as i64)
                            .remainder_scalar(height as i64);
                        let w = indices.remainder_scalar(width as i64);
                        w.mul_scalar((depth * height) as i64) + h.mul_scalar(depth as i64) + d
                    } + volumes,
                )
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn test_max_pool3d_int8() {
        let code = codegen_forward_default(&typed_node("input", DType::I8, false, 0));
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5, Int>) -> Tensor<5, Int> {
            let output = {
                let (padding, input) = (
                    [(1usize, 0usize), (0usize, 1usize), (1usize, 1usize)],
                    input,
                );
                burn::tensor::module::max_pool3d(
                        input.float().cast(burn::tensor::DType::F32),
                        burn::tensor::ops::MaxPoolOptions::new([2, 3, 3])
                            .with_stride([2, 2, 2])
                            .with_padding_pairs(padding)
                            .with_dilation([1, 2, 1])
                            .with_ceil_mode(true),
                    )
                    .int()
                    .cast(burn::tensor::DType::I8)
            };
            output
        }
        ");
    }

    #[test]
    fn test_max_pool3d_uint8_indices() {
        let code = codegen_forward_default(&typed_node("input", DType::U8, true, 0));
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5, Int>) -> (Tensor<5, Int>, Tensor<5, Int>) {
            let (output, indices) = {
                let (padding, input) = (
                    [(1usize, 0usize), (0usize, 1usize), (1usize, 1usize)],
                    input,
                );
                let [batch, channels, depth, height, width] = input.dims();
                let (values, indices) = burn::tensor::module::max_pool3d_with_indices(
                    input.cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32),
                    burn::tensor::ops::MaxPoolOptions::new([2, 3, 3])
                        .with_stride([2, 2, 2])
                        .with_padding_pairs(padding)
                        .with_dilation([1, 2, 1])
                        .with_ceil_mode(true),
                );
                let indices = indices.cast(burn::tensor::DType::I64);
                let volumes = Tensor::<
                    1,
                    Int,
                >::arange(0..(batch * channels) as i64, (&self.device, burn::tensor::DType::I64))
                    .mul_scalar((depth * height * width) as i64)
                    .reshape([batch, channels, 1, 1, 1]);
                (values.int().cast(burn::tensor::DType::U8), indices + volumes)
            };
            (output, indices)
        }
        ");
    }

    #[test]
    fn input_named_like_its_temporaries_is_accepted() {
        let temporaries = [
            "padding", "input", "batch", "channels", "depth", "height", "width", "values",
            "indices", "volumes", "d", "h", "w",
        ];
        for temporary in temporaries {
            for (dtype, indices, storage_order) in [
                (DType::F32, false, 0),
                (DType::F32, true, 0),
                (DType::F32, true, 1),
                (DType::U8, false, 0),
                (DType::I8, true, 1),
            ] {
                assert_eq!(
                    shadow_check_result(&typed_node(temporary, dtype, indices, storage_order)),
                    Ok(()),
                    "input named {temporary}, {dtype:?}, indices {indices}, storage_order {storage_order}"
                );
            }
        }
    }
}
