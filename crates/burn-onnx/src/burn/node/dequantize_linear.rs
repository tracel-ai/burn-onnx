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

        let scale_expr = if scale_arg.ty.elem_type() == out_dtype {
            quote! { #scale }
        } else if scale_arg.ty.elem_type().is_float() {
            quote! { (#scale).cast(#out_dtype_tokens) }
        } else {
            quote! { (#scale).float().cast(#out_dtype_tokens) }
        };

        let x_rank = x_arg.ty.rank();
        let scale_rank = scale_arg.ty.rank();
        let scale_expr = align_rhs_for_lhs_rank(scale_expr, x_rank, scale_rank, self.config.axis);

        let centered_expr = if let Some(zp_arg) = self.inputs.get(2) {
            let zp = scope.arg(zp_arg);
            let zp_expr = if zp_arg.ty.elem_type() == out_dtype {
                quote! { #zp }
            } else if zp_arg.ty.elem_type().is_float() {
                quote! { (#zp).cast(#out_dtype_tokens) }
            } else if zp_arg.ty.elem_type().is_uint() {
                quote! { (#zp).cast(burn::tensor::DType::I32).float().cast(#out_dtype_tokens) }
            } else {
                quote! { (#zp).float().cast(#out_dtype_tokens) }
            };
            let zp_expr =
                align_rhs_for_lhs_rank(zp_expr, x_rank, zp_arg.ty.rank(), self.config.axis);
            quote! { (#x_expr).sub(#zp_expr) }
        } else {
            x_expr
        };

        quote! {
            let #output = (#centered_expr).mul(#scale_expr);
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
        pub fn forward(&self, x: Tensor<B, 2, Int>, x_scale: Tensor<B, 0>) -> Tensor<B, 2> {
            let y = ((x).cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32))
                .mul((x_scale).unsqueeze_dims(&[0isize, 1isize]));
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
            x: Tensor<B, 2, Int>,
            x_scale: Tensor<B, 0>,
            x_zero_point: Tensor<B, 0, Int>,
        ) -> Tensor<B, 2> {
            let y = (((x).cast(burn::tensor::DType::I32).float().cast(burn::tensor::DType::F32))
                .sub(
                    ((x_zero_point)
                        .cast(burn::tensor::DType::I32)
                        .float()
                        .cast(burn::tensor::DType::F32))
                        .unsqueeze_dims(&[0isize, 1isize]),
                ))
                .mul((x_scale).unsqueeze_dims(&[0isize, 1isize]));
            y
        }
        ");
    }
}
