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

        let target_dtype = self.outputs.first().unwrap().ty.elem_type();
        let target_dtype_tokens = target_dtype.to_tokens();

        let x_expr = if x_arg.ty.elem_type().is_float() {
            quote! { (#x).cast(burn::tensor::DType::F32) }
        } else if x_arg.ty.elem_type().is_uint() {
            quote! { (#x).cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32) }
        } else {
            quote! { (#x).float().cast(burn::tensor::DType::F32) }
        };

        let scale_expr = if scale_arg.ty.elem_type().is_float() {
            quote! { (#scale).cast(burn::tensor::DType::F32) }
        } else {
            quote! { (#scale).float().cast(burn::tensor::DType::F32) }
        };

        let x_rank = x_arg.ty.rank();
        let scale_rank = scale_arg.ty.rank();
        let scale_expr = align_rhs_for_lhs_rank(scale_expr, x_rank, scale_rank, self.config.axis);

        let with_zero_point = if let Some(zp_arg) = self.inputs.get(2) {
            let zp = scope.arg(zp_arg);
            let zp_expr = if zp_arg.ty.elem_type().is_float() {
                quote! { (#zp).cast(burn::tensor::DType::F32) }
            } else if zp_arg.ty.elem_type().is_uint() {
                quote! { (#zp).cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32) }
            } else {
                quote! { (#zp).float().cast(burn::tensor::DType::F32) }
            };
            let zp_expr =
                align_rhs_for_lhs_rank(zp_expr, x_rank, zp_arg.ty.rank(), self.config.axis);
            quote! { ((#x_expr).div(#scale_expr)).round().add(#zp_expr) }
        } else {
            quote! { ((#x_expr).div(#scale_expr)).round() }
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

        quote! {
            let #output = (#clamped_expr).int().cast(#target_dtype_tokens);
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
        pub fn forward(&self, x: Tensor<B, 2>, y_scale: Tensor<B, 0>) -> Tensor<B, 2, Int> {
            let y = (((((x).cast(burn::tensor::DType::F32))
                .div(
                    ((y_scale).cast(burn::tensor::DType::F32)).unsqueeze_dims(&[0isize, 1isize]),
                ))
                .round())
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
            x: Tensor<B, 2>,
            y_scale: Tensor<B, 0>,
            y_zero_point: Tensor<B, 0, Int>,
        ) -> Tensor<B, 2, Int> {
            let y = (((((x).cast(burn::tensor::DType::F32))
                .div(
                    ((y_scale).cast(burn::tensor::DType::F32)).unsqueeze_dims(&[0isize, 1isize]),
                ))
                .round()
                .add(
                    ((y_zero_point)
                        .cast(burn::tensor::DType::I32)
                        .float()
                        .cast(burn::tensor::DType::F32))
                        .unsqueeze_dims(&[0isize, 1isize]),
                ))
                .clamp(0f32, 255f32))
                .int()
                .cast(burn::tensor::DType::U8);
            y
        }
        ");
    }
}
