use super::broadcast_helpers::align_rhs_for_lhs_rank;
use super::prelude::*;

impl NodeCodegen for onnx_ir::dequantize_linear::DequantizeLinearNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let x_arg = self.inputs.first().unwrap();
        let scale_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let x = scope.arg(x_arg);
        let scale = scope.arg(scale_arg);
        let zp_arg = self.inputs.get(2).filter(|arg| !arg.is_optional());
        let zp = zp_arg.map(|arg| scope.arg(arg));
        // Capture dimensions and every graph value together before local bindings
        // can shadow an input or a shared parameter can consume the data tensor.
        let (prepare, x, scale, zp) = if self.config.block_size.is_some_and(|size| size > 0) {
            if let Some(zp) = zp {
                (
                    quote! { let (input_dims, input, scale, zero_point) = ((#x).dims(), #x, #scale, #zp); },
                    quote! { input },
                    quote! { scale },
                    Some(quote! { zero_point }),
                )
            } else {
                (
                    quote! { let (input_dims, input, scale) = ((#x).dims(), #x, #scale); },
                    quote! { input },
                    quote! { scale },
                    None,
                )
            }
        } else {
            (quote! {}, x, scale, zp)
        };
        let input_shape_source = quote! { input_dims };

        let out_dtype = self.outputs.first().unwrap().ty.elem_type();
        let out_dtype_tokens = out_dtype.to_tokens();

        let x_expr = if x_arg.ty.elem_type() == out_dtype {
            quote! { #x }
        } else if x_arg.ty.elem_type().is_float() {
            quote! { (#x).cast(#out_dtype_tokens) }
        } else if x_arg.ty.elem_type().is_uint() {
            quote! { (#x).cast(burn::tensor::DType::I32).float().cast(#out_dtype_tokens) }
        } else {
            quote! { (#x).float().cast(#out_dtype_tokens) }
        };

        let scale_expr = if matches!(scale_arg.ty, ArgType::ScalarNative(_)) {
            quote! { burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::from([burn::tensor::ElementConversion::elem::<f64>(#scale)]), (self.device(), #out_dtype_tokens)) }
        } else if scale_arg.ty.elem_type() == out_dtype {
            quote! { #scale }
        } else if scale_arg.ty.elem_type().is_float() {
            quote! { (#scale).cast(#out_dtype_tokens) }
        } else {
            quote! { (#scale).float().cast(#out_dtype_tokens) }
        };

        let x_rank = x_arg.ty.rank();
        let scale_rank = scale_arg.ty.rank().max(1);
        let scale_expr = if let Some(block) = self.config.block_size.filter(|b| *b > 0) {
            super::quantization::expand_blocks(
                scale_expr,
                &input_shape_source,
                x_rank,
                self.config.axis.unwrap_or(1),
                block,
            )
        } else {
            align_rhs_for_lhs_rank(scale_expr, x_rank, scale_rank, self.config.axis)
        };

        let centered_expr = if let (Some(zp_arg), Some(zp)) = (zp_arg, zp) {
            let integer_centering = !x_arg.ty.elem_type().is_float();
            // I32 exactly centers 8/16-bit values and works on more backends than I64.
            let center_dtype = if x_arg.ty.elem_type().size() > 2 {
                DType::I64
            } else {
                DType::I32
            }
            .to_tokens();
            let zp_expr = if integer_centering {
                if matches!(zp_arg.ty, ArgType::ScalarNative(_)) {
                    quote! { burn::tensor::Tensor::<1, burn::tensor::Int>::from_data(burn::tensor::TensorData::from([burn::tensor::ElementConversion::elem::<i64>(#zp)]), (self.device(), #center_dtype)) }
                } else {
                    quote! { (#zp).cast(#center_dtype) }
                }
            } else if matches!(zp_arg.ty, ArgType::ScalarNative(_)) {
                let dtype = out_dtype_tokens.clone();
                quote! { burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::from([burn::tensor::ElementConversion::elem::<f64>(#zp)]), (self.device(), #dtype)) }
            } else if zp_arg.ty.elem_type() == out_dtype {
                quote! { #zp }
            } else if zp_arg.ty.elem_type().is_float() {
                quote! { (#zp).cast(#out_dtype_tokens) }
            } else if zp_arg.ty.elem_type().is_uint() {
                quote! { (#zp).cast(burn::tensor::DType::I32).float().cast(#out_dtype_tokens) }
            } else {
                quote! { (#zp).float().cast(#out_dtype_tokens) }
            };
            let zp_expr = if let Some(block) = self.config.block_size.filter(|b| *b > 0) {
                super::quantization::expand_blocks(
                    zp_expr,
                    &input_shape_source,
                    x_rank,
                    self.config.axis.unwrap_or(1),
                    block,
                )
            } else {
                align_rhs_for_lhs_rank(zp_expr, x_rank, zp_arg.ty.rank().max(1), self.config.axis)
            };
            if integer_centering {
                quote! { (#x).cast(#center_dtype).sub(#zp_expr).float().cast(#out_dtype_tokens) }
            } else {
                quote! { (#x_expr).sub(#zp_expr) }
            }
        } else {
            x_expr
        };

        let value = quote! { (#centered_expr).mul(#scale_expr) };
        if prepare.is_empty() {
            quote! { let #output = #value; }
        } else {
            quote! { let #output = { #prepare #value }; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::dequantize_linear::{
        DequantizeLinearConfig, DequantizeLinearNode, DequantizeLinearNodeBuilder,
    };

    fn create_node(name: &str, with_zero_point: bool) -> DequantizeLinearNode {
        let mut builder = DequantizeLinearNodeBuilder::new(name)
            .input_tensor("x", 2, DType::U8)
            .input_tensor("x_scale", 0, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(DequantizeLinearConfig::default());

        if with_zero_point {
            builder = builder.input_tensor("x_zero_point", 0, DType::U8);
        }

        builder.build()
    }

    #[test]
    fn test_dequantize_linear_forward_without_zero_point() {
        let node = create_node("dq", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<2, Int>, x_scale: Tensor<0>) -> Tensor<2> {
            let y = ((x).cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32))
                .mul((x_scale).unsqueeze_dims(&[0isize]));
            y
        }
        ");
    }

    #[test]
    fn test_dequantize_linear_forward_with_zero_point() {
        let node = create_node("dq", true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            x: Tensor<2, Int>,
            x_scale: Tensor<0>,
            x_zero_point: Tensor<0, Int>,
        ) -> Tensor<2> {
            let y = ((x)
                .cast(burn::tensor::DType::I32)
                .sub(((x_zero_point).cast(burn::tensor::DType::I32)).unsqueeze_dims(&[0isize]))
                .float()
                .cast(burn::tensor::DType::F32))
                .mul((x_scale).unsqueeze_dims(&[0isize]));
            y
        }
        ");
    }
    #[test]
    fn native_parameters() {
        let node = DequantizeLinearNodeBuilder::new("q")
            .input_tensor("x", 2, DType::U8)
            .input_scalar("scale", DType::F32)
            .input_scalar("zero", DType::U8)
            .output_tensor("y", 2, DType::F32)
            .config(DequantizeLinearConfig::default())
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, x: Tensor<2, Int>, scale: f32, zero: u8) -> Tensor<2> {
            let y = ((x)
                .cast(burn::tensor::DType::I32)
                .sub(
                    (burn::tensor::Tensor::<
                        1,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::from([
                            burn::tensor::ElementConversion::elem::<i64>(zero),
                        ]),
                        (self.device(), burn::tensor::DType::I32),
                    ))
                        .unsqueeze_dims(&[0isize]),
                )
                .float()
                .cast(burn::tensor::DType::F32))
                .mul(
                    (burn::tensor::Tensor::<
                        1,
                    >::from_data(
                        burn::tensor::TensorData::from([
                            burn::tensor::ElementConversion::elem::<f64>(scale),
                        ]),
                        (self.device(), burn::tensor::DType::F32),
                    ))
                        .unsqueeze_dims(&[0isize]),
                );
            y
        }
        ");
    }
    #[test]
    fn blocked_parameters() {
        let node = DequantizeLinearNodeBuilder::new("q")
            .input_tensor("x", 2, DType::U8)
            .input_tensor("scale", 2, DType::F32)
            .input_tensor("zero", 2, DType::U8)
            .output_tensor("y", 2, DType::F32)
            .config(DequantizeLinearConfig {
                axis: Some(-1),
                block_size: Some(2),
                ..Default::default()
            })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(
            &self,
            x: Tensor<2, Int>,
            scale: Tensor<2>,
            zero: Tensor<2, Int>,
        ) -> Tensor<2> {
            let y = {
                let (input_dims, input, scale, zero_point) = ((x).dims(), x, scale, zero);
                ((input)
                    .cast(burn::tensor::DType::I32)
                    .sub({
                        let (parameter, axis_size) = (
                            (zero_point).cast(burn::tensor::DType::I32),
                            input_dims[1usize],
                        );
                        let mut shape = parameter.dims();
                        shape[1usize] *= 2usize;
                        parameter
                            .unsqueeze_dim::<3usize>(2usize)
                            .repeat_dim(2usize, 2usize)
                            .reshape::<2usize, _>(shape)
                            .slice_dim(1usize, 0..axis_size)
                    })
                    .float()
                    .cast(burn::tensor::DType::F32))
                    .mul({
                        let (parameter, axis_size) = (scale, input_dims[1usize]);
                        let mut shape = parameter.dims();
                        shape[1usize] *= 2usize;
                        parameter
                            .unsqueeze_dim::<3usize>(2usize)
                            .repeat_dim(2usize, 2usize)
                            .reshape::<2usize, _>(shape)
                            .slice_dim(1usize, 0..axis_size)
                    })
            };
            y
        }
        ");
    }
}
