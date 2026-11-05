use super::broadcast_helpers::align_rhs_for_lhs_rank;
use super::prelude::*;

impl NodeCodegen for onnx_ir::quantize_linear::QuantizeLinearNode {
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

        let calculation_dtype = self
            .config
            .precision
            .unwrap_or_else(|| {
                let dtype = scale_arg.ty.elem_type();
                if dtype.is_float() { dtype } else { DType::F32 }
            })
            .to_tokens();
        let target_dtype = self.outputs.first().unwrap().ty.elem_type();
        let target_dtype_tokens = target_dtype.to_tokens();

        let x_expr = if x_arg.ty.elem_type().is_float() {
            quote! { (#x).cast(#calculation_dtype) }
        } else if x_arg.ty.elem_type().is_uint() {
            quote! { (#x).cast(burn::tensor::DType::I32).float().cast(#calculation_dtype) }
        } else {
            quote! { (#x).float().cast(#calculation_dtype) }
        };

        let scale_expr = if matches!(scale_arg.ty, ArgType::ScalarNative(_)) {
            quote! { burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::from([burn::tensor::ElementConversion::elem::<f64>(#scale)]), (self.device(), #calculation_dtype)) }
        } else if scale_arg.ty.elem_type().is_float() {
            quote! { (#scale).cast(#calculation_dtype) }
        } else {
            quote! { (#scale).float().cast(#calculation_dtype) }
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

        let with_zero_point = if let (Some(zp_arg), Some(zp)) = (zp_arg, zp) {
            let zp_expr = if matches!(zp_arg.ty, ArgType::ScalarNative(_)) {
                let dtype = quote! { burn::tensor::DType::F32 };
                quote! { burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::from([burn::tensor::ElementConversion::elem::<f64>(#zp)]), (self.device(), #dtype)) }
            } else if zp_arg.ty.elem_type().is_float() {
                quote! { (#zp).cast(burn::tensor::DType::F32) }
            } else if zp_arg.ty.elem_type().is_uint() {
                quote! { (#zp).cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32) }
            } else {
                quote! { (#zp).float().cast(burn::tensor::DType::F32) }
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
            quote! { ((#x_expr).div(#scale_expr)).round().cast(burn::tensor::DType::F32).add(#zp_expr) }
        } else {
            quote! { ((#x_expr).div(#scale_expr)).round().cast(burn::tensor::DType::F32) }
        };

        let clamped_expr = match target_dtype {
            DType::U8 => quote! { (#with_zero_point).clamp(0f32, 255f32) },
            DType::I8 => quote! { (#with_zero_point).clamp(-128f32, 127f32) },
            DType::U16 => quote! { (#with_zero_point).clamp(0f32, 65535f32) },
            DType::I16 => quote! { (#with_zero_point).clamp(-32768f32, 32767f32) },
            _ => panic!(
                "QuantizeLinear output dtype {:?} not supported",
                target_dtype
            ),
        };

        let value = quote! { (#clamped_expr).int().cast(#target_dtype_tokens) };
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
    use onnx_ir::quantize_linear::{
        QuantizeLinearConfig, QuantizeLinearNode, QuantizeLinearNodeBuilder,
    };

    fn create_node(name: &str, with_zero_point: bool) -> QuantizeLinearNode {
        let mut builder = QuantizeLinearNodeBuilder::new(name)
            .input_tensor("x", 2, DType::F32)
            .input_tensor("y_scale", 0, DType::F32)
            .output_tensor("y", 2, DType::U8)
            .config(QuantizeLinearConfig::default());

        if with_zero_point {
            builder = builder.input_tensor("y_zero_point", 0, DType::U8);
        }

        builder.build()
    }

    #[test]
    fn test_quantize_linear_forward_without_zero_point() {
        let node = create_node("q", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<2>, y_scale: Tensor<0>) -> Tensor<2, Int> {
            let y = (((((x).cast(burn::tensor::DType::F32))
                .div(((y_scale).cast(burn::tensor::DType::F32)).unsqueeze_dims(&[0isize])))
                .round()
                .cast(burn::tensor::DType::F32))
                .clamp(0f32, 255f32))
                .int()
                .cast(burn::tensor::DType::U8);
            y
        }
        ");
    }

    #[test]
    fn test_quantize_linear_forward_with_zero_point() {
        let node = create_node("q", true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            x: Tensor<2>,
            y_scale: Tensor<0>,
            y_zero_point: Tensor<0, Int>,
        ) -> Tensor<2, Int> {
            let y = (((((x).cast(burn::tensor::DType::F32))
                .div(((y_scale).cast(burn::tensor::DType::F32)).unsqueeze_dims(&[0isize])))
                .round()
                .cast(burn::tensor::DType::F32)
                .add(
                    ((y_zero_point)
                        .cast(burn::tensor::DType::I32)
                        .float()
                        .cast(burn::tensor::DType::F32))
                        .unsqueeze_dims(&[0isize]),
                ))
                .clamp(0f32, 255f32))
                .int()
                .cast(burn::tensor::DType::U8);
            y
        }
        ");
    }
    #[test]
    fn native_parameters() {
        let node = QuantizeLinearNodeBuilder::new("q")
            .input_tensor("x", 2, DType::F32)
            .input_scalar("scale", DType::F32)
            .input_scalar("zero", DType::U8)
            .output_tensor("y", 2, DType::U8)
            .config(QuantizeLinearConfig::default())
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, x: Tensor<2>, scale: f32, zero: u8) -> Tensor<2, Int> {
            let y = (((((x).cast(burn::tensor::DType::F32))
                .div(
                    (burn::tensor::Tensor::<
                        1,
                    >::from_data(
                        burn::tensor::TensorData::from([
                            burn::tensor::ElementConversion::elem::<f64>(scale),
                        ]),
                        (self.device(), burn::tensor::DType::F32),
                    ))
                        .unsqueeze_dims(&[0isize]),
                ))
                .round()
                .cast(burn::tensor::DType::F32)
                .add(
                    (burn::tensor::Tensor::<
                        1,
                    >::from_data(
                        burn::tensor::TensorData::from([
                            burn::tensor::ElementConversion::elem::<f64>(zero),
                        ]),
                        (self.device(), burn::tensor::DType::F32),
                    ))
                        .unsqueeze_dims(&[0isize]),
                ))
                .clamp(0f32, 255f32))
                .int()
                .cast(burn::tensor::DType::U8);
            y
        }
        ");
    }
    #[test]
    fn blocked_parameters() {
        let node = QuantizeLinearNodeBuilder::new("q")
            .input_tensor("x", 2, DType::F32)
            .input_tensor("scale", 2, DType::F32)
            .input_tensor("zero", 2, DType::U8)
            .output_tensor("y", 2, DType::U8)
            .config(QuantizeLinearConfig {
                axis: Some(-1),
                block_size: Some(2),
                ..Default::default()
            })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(
            &self,
            x: Tensor<2>,
            scale: Tensor<2>,
            zero: Tensor<2, Int>,
        ) -> Tensor<2, Int> {
            let y = {
                let (input_dims, input, scale, zero_point) = ((x).dims(), x, scale, zero);
                (((((input).cast(burn::tensor::DType::F32))
                    .div({
                        let (parameter, axis_size) = (
                            (scale).cast(burn::tensor::DType::F32),
                            input_dims[1usize],
                        );
                        let mut shape = parameter.dims();
                        shape[1usize] *= 2usize;
                        parameter
                            .unsqueeze_dim::<3usize>(2usize)
                            .repeat_dim(2usize, 2usize)
                            .reshape::<2usize, _>(shape)
                            .slice_dim(1usize, 0..axis_size)
                    }))
                    .round()
                    .cast(burn::tensor::DType::F32)
                    .add({
                        let (parameter, axis_size) = (
                            (zero_point)
                                .cast(burn::tensor::DType::I32)
                                .float()
                                .cast(burn::tensor::DType::F32),
                            input_dims[1usize],
                        );
                        let mut shape = parameter.dims();
                        shape[1usize] *= 2usize;
                        parameter
                            .unsqueeze_dim::<3usize>(2usize)
                            .repeat_dim(2usize, 2usize)
                            .reshape::<2usize, _>(shape)
                            .slice_dim(1usize, 0..axis_size)
                    }))
                    .clamp(0f32, 255f32))
                    .int()
                    .cast(burn::tensor::DType::U8)
            };
            y
        }
        ");
    }
}
