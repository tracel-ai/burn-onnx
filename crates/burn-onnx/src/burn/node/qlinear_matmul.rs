use super::prelude::*;

impl NodeCodegen for onnx_ir::qlinear_matmul::QLinearMatMulNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // ASSUMPTIONS (per the ONNX spec)
        // 1. Scale and zero point input for a given operand have the same shape.
        // 2. Tensor operands have the same rank.
        let a_arg = self.inputs.first().unwrap();
        let a_scale_arg = self.inputs.get(1).unwrap();
        let a_zero_point_arg = self.inputs.get(2).unwrap();
        let b_arg = self.inputs.get(3).unwrap();
        let b_scale_arg = self.inputs.get(4).unwrap();
        let b_zero_point_arg = self.inputs.get(5).unwrap();
        let y_scale_arg = self.inputs.get(6).unwrap();
        let y_zero_point_arg = self.inputs.get(7).unwrap();

        let output = arg_to_ident(self.outputs.first().unwrap());

        let a = scope.arg(a_arg);
        let a_scale = scope.arg(a_scale_arg);
        let a_zero_point = scope.arg(a_zero_point_arg);
        let b = scope.arg(b_arg);
        let b_scale = scope.arg(b_scale_arg);
        let b_zero_point = scope.arg(b_zero_point_arg);
        let y_scale = scope.arg(y_scale_arg);
        let y_zero_point = scope.arg(y_zero_point_arg);

        let output_elem_type = y_zero_point_arg.ty.elem_type();
        let output_dtype = output_elem_type.to_tokens();
        let clamp_expr = match output_elem_type {
            DType::U8 => quote! { .clamp(0f32, 255f32) },
            DType::I8 => quote! { .clamp(-128f32, 127f32) },
            _ => quote! {},
        };

        if matches!(a_arg.ty.elem_type(), DType::QFloat(..))
            || matches!(b_arg.ty.elem_type(), DType::QFloat(..))
            || matches!(y_zero_point_arg.ty.elem_type(), DType::QFloat(..))
        {
            panic!("Quantized floats are not supported in `burn`")
        }

        let reshape_a_scale_and_zp = reshape_scale_and_zp(a_scale_arg, &a, &a_scale, &a_zero_point);
        let reshape_b_scale_and_zp = reshape_scale_and_zp(b_scale_arg, &b, &b_scale, &b_zero_point);

        // Generate y reshape code by passing a TokenStream for `output_tensor`.
        // This will be inserted after `output_tensor` is created in the quote block.
        let output_tensor_ts = quote! { output_tensor };
        let reshape_y_scale_and_zp =
            reshape_scale_and_zp(y_scale_arg, &output_tensor_ts, &y_scale, &y_zero_point);

        // Convert matmul operands and zero_points into floats
        // NOTE: A faster path can be achieved if both `a` and `b` are int types,
        //       by performing matmul in I32 before applying the scale inputs.
        //       See https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_qlinear_matmul.py
        let a_float = to_float(a_arg, a);
        let a_zero_point_float = to_float(a_zero_point_arg, a_zero_point);
        let b_float = to_float(b_arg, b);
        let b_zero_point_float = to_float(b_zero_point_arg, b_zero_point);
        let y_zero_point_float = to_float(y_zero_point_arg, y_zero_point);

        quote! {
            // Dequantize inputs
            #reshape_a_scale_and_zp
            #reshape_b_scale_and_zp
            let a_dequantized = #a_scale * (#a_float - #a_zero_point_float);
            let b_dequantized = #b_scale * (#b_float - #b_zero_point_float);

            let output_tensor = a_dequantized.matmul(b_dequantized);

            // Quantize output
            #reshape_y_scale_and_zp
            let #output = (output_tensor / #y_scale).round();
            let #output = (#output + #y_zero_point_float)#clamp_expr.int().cast(#output_dtype);
        }
    }
}

/// Cast the tensor and zero points to float
fn to_float(arg: &Argument, token_stream: TokenStream) -> TokenStream {
    let needs_cast = arg.ty.elem_type().is_int() || arg.ty.elem_type().is_uint();
    if !needs_cast {
        token_stream
    } else if arg.ty.is_scalar() {
        quote! { (#token_stream as f32) }
    } else {
        quote! { #token_stream.float() }
    }
}

fn reshape_scale_and_zp(
    scale_arg: &Argument,
    tensor: &TokenStream,
    scale: &TokenStream,
    zero_point: &TokenStream,
) -> TokenStream {
    // There are three possible quantization cases, based on the scale and zero point shapes:
    //   1. Scalar; per-tensor quantization
    //   2. Vectors; row/column-based quantization
    //   3. N-D tensors; higher-dim row/column-based quantization
    // Reshape the scale and zero point, if necessary.
    if scale_arg.ty.is_scalar() {
        // Case 1: All zero point and inputs are scalars
        // Reshaping is not required.
        quote! {}
    } else if scale_arg.ty.rank() == 1 {
        // Case 2: Scale and zero point are vectors. Infer whether they are row or column vectors and expand appropriate.
        quote! {
            let expansion_dim = if #scale.dims()[0] == #tensor.dims()[0] { 1 } else { 0 };
            let #scale = #scale.unsqueeze_dim(expansion_dim);
            let #zero_point = #zero_point.unsqueeze_dim(expansion_dim);
        }
    } else {
        // Case 3: Scale and zero_point have the same rank as their operands. Either the last or second-to-last dimension has a size of 1.
        // Reshaping is not required.
        quote! {}
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{
        DType,
        quantization::{QuantScheme, QuantValue},
    };
    use insta::assert_snapshot;
    use onnx_ir::qlinear_matmul::QLinearMatMulNodeBuilder;

    #[test]
    fn test_qlinear_matmul_case_1_scalar() {
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor("a", 2, DType::I8)
            .input_scalar("a_scale", DType::F32)
            .input_scalar("a_zero_point", DType::I8)
            .input_tensor("b", 2, DType::I8)
            .input_scalar("b_scale", DType::F32)
            .input_scalar("b_zero_point", DType::I8)
            .input_scalar("y_scale", DType::F32)
            .input_scalar("y_zero_point", DType::I8)
            .output_tensor("y", 2, DType::I8)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            a_scale: f32,
            a_zero_point: i8,
            b: Tensor<B, 2, Int>,
            b_scale: f32,
            b_zero_point: i8,
            y_scale: f32,
            y_zero_point: i8,
        ) -> Tensor<B, 2, Int> {
            let a_dequantized = a_scale
                * (a.cast(burn::tensor::DType::I32).float() - (a_zero_point as f32));
            let b_dequantized = b_scale
                * (b.cast(burn::tensor::DType::I32).float() - (b_zero_point as f32));
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = (output_tensor / y_scale).round();
            let y = (y + (y_zero_point as f32))
                .clamp(-128f32, 127f32)
                .int()
                .cast(burn::tensor::DType::I8);
            y
        }
        ");
    }

    #[test]
    fn test_qlinear_matmul_case_1_scalar_different_output_dtype() {
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor("a", 2, DType::I8)
            .input_scalar("a_scale", DType::F32)
            .input_scalar("a_zero_point", DType::I8)
            .input_tensor("b", 2, DType::I8)
            .input_scalar("b_scale", DType::F32)
            .input_scalar("b_zero_point", DType::I8)
            .input_scalar("y_scale", DType::F32)
            .input_scalar("y_zero_point", DType::BF16)
            .output_tensor("y", 2, DType::BF16) // Different from `a` and `b`
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            a_scale: f32,
            a_zero_point: i8,
            b: Tensor<B, 2, Int>,
            b_scale: f32,
            b_zero_point: i8,
            y_scale: f32,
            y_zero_point: half::bf16,
        ) -> Tensor<B, 2> {
            let a_dequantized = a_scale
                * (a.cast(burn::tensor::DType::I32).float() - (a_zero_point as f32));
            let b_dequantized = b_scale
                * (b.cast(burn::tensor::DType::I32).float() - (b_zero_point as f32));
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = (output_tensor / y_scale).round();
            let y = (y + y_zero_point).int().cast(burn::tensor::DType::BF16);
            y
        }
        ");
    }

    #[test]
    #[should_panic(expected = "Quantized floats are not supported in `burn`")]
    fn test_qlinear_matmul_case_1_f8_dtype() {
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor(
                "a",
                2,
                DType::QFloat(QuantScheme::default().with_value(QuantValue::E5M2)),
            )
            .input_scalar("a_scale", DType::F32)
            .input_scalar(
                "a_zero_point",
                DType::QFloat(QuantScheme::default().with_value(QuantValue::E5M2)),
            )
            .input_tensor(
                "b",
                2,
                DType::QFloat(QuantScheme::default().with_value(QuantValue::E5M2)),
            )
            .input_scalar("b_scale", DType::F32)
            .input_scalar(
                "b_zero_point",
                DType::QFloat(QuantScheme::default().with_value(QuantValue::E5M2)),
            )
            .input_scalar("y_scale", DType::F32)
            .input_tensor("y_zero_point", 0, DType::I8)
            .output_tensor("y", 2, DType::I8)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"");
    }

    #[test]
    fn test_qlinear_matmul_case_2_vector() {
        // Case 2: Vector scale/zero_points (rank 1) with 2D tensor operands
        // Vectors are expanded to match operand rank
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor("a", 2, DType::I8)
            .input_tensor("a_scale", 1, DType::F32)
            .input_tensor("a_zero_point", 1, DType::I8)
            .input_tensor("b", 2, DType::I8)
            .input_tensor("b_scale", 1, DType::F32)
            .input_tensor("b_zero_point", 1, DType::I8)
            .input_tensor("y_scale", 1, DType::F32)
            .input_tensor("y_zero_point", 1, DType::I8)
            .output_tensor("y", 2, DType::I8)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            a_scale: Tensor<B, 1>,
            a_zero_point: Tensor<B, 1, Int>,
            b: Tensor<B, 2, Int>,
            b_scale: Tensor<B, 1>,
            b_zero_point: Tensor<B, 1, Int>,
            y_scale: Tensor<B, 1>,
            y_zero_point: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2, Int> {
            let expansion_dim = if a_scale.dims()[0] == a.dims()[0] { 1 } else { 0 };
            let a_scale = a_scale.unsqueeze_dim(expansion_dim);
            let a_zero_point = a_zero_point.unsqueeze_dim(expansion_dim);
            let expansion_dim = if b_scale.dims()[0] == b.dims()[0] { 1 } else { 0 };
            let b_scale = b_scale.unsqueeze_dim(expansion_dim);
            let b_zero_point = b_zero_point.unsqueeze_dim(expansion_dim);
            let a_dequantized = a_scale
                * (a.cast(burn::tensor::DType::I32).float()
                    - a_zero_point.cast(burn::tensor::DType::I32).float());
            let b_dequantized = b_scale
                * (b.cast(burn::tensor::DType::I32).float()
                    - b_zero_point.cast(burn::tensor::DType::I32).float());
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let expansion_dim = if y_scale.dims()[0] == output_tensor.dims()[0] { 1 } else { 0 };
            let y_scale = y_scale.unsqueeze_dim(expansion_dim);
            let y_zero_point = y_zero_point.unsqueeze_dim(expansion_dim);
            let y = (output_tensor / y_scale).round();
            let y = (y + y_zero_point.cast(burn::tensor::DType::I32).float())
                .clamp(-128f32, 127f32)
                .int()
                .cast(burn::tensor::DType::I8);
            y
        }
        ");
    }

    #[test]
    fn test_qlinear_matmul_case_3_nd_tensor() {
        // Case 3: N-D scale/zero_points with same rank as tensor operands (3D in this case)
        // No reshaping needed; scale/zero_points are already the correct shape
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor("a", 3, DType::I8)
            .input_tensor("a_scale", 3, DType::F32)
            .input_tensor("a_zero_point", 3, DType::I8)
            .input_tensor("b", 3, DType::I8)
            .input_tensor("b_scale", 3, DType::F32)
            .input_tensor("b_zero_point", 3, DType::I8)
            .input_tensor("y_scale", 3, DType::F32)
            .input_tensor("y_zero_point", 3, DType::I8)
            .output_tensor("y", 3, DType::I8)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 3, Int>,
            a_scale: Tensor<B, 3>,
            a_zero_point: Tensor<B, 3, Int>,
            b: Tensor<B, 3, Int>,
            b_scale: Tensor<B, 3>,
            b_zero_point: Tensor<B, 3, Int>,
            y_scale: Tensor<B, 3>,
            y_zero_point: Tensor<B, 3, Int>,
        ) -> Tensor<B, 3, Int> {
            let a_dequantized = a_scale
                * (a.cast(burn::tensor::DType::I32).float()
                    - a_zero_point.cast(burn::tensor::DType::I32).float());
            let b_dequantized = b_scale
                * (b.cast(burn::tensor::DType::I32).float()
                    - b_zero_point.cast(burn::tensor::DType::I32).float());
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = (output_tensor / y_scale).round();
            let y = (y + y_zero_point.cast(burn::tensor::DType::I32).float())
                .clamp(-128f32, 127f32)
                .int()
                .cast(burn::tensor::DType::I8);
            y
        }
        ");
    }
}
