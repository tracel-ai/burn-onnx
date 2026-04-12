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

        let output_dtype = y_zero_point_arg.ty.elem_type().to_tokens();

        // There are three possible quantization cases, based on the scale and zero point shapes:
        //   1. Scalar; per-tensor quantization
        //   2. Vectors; row/column-based quantization
        //   3. N-D tensors; higher-dim row/column-based quantization
        // Reshape the scale and zero points, if necessary.
        let reshape_code = if a_scale_arg.ty.is_scalar() {
            // Case 1: All zero points and inputs are scalars
            // Reshaping is not required.
            quote! {}
        } else if a_scale_arg.ty.rank() == 1 {
            // Case 2: Scale and zero points are vectors. Infer whether they are row or column vectors and expand appropriate.
            quote! {
                let a_expansion_dim = if #a_scale.len() == #a.dims()[0] { 1 } else { 0 };
                let a_scale = #a_scale.unsqueeze_dims(a_expansion_dim);
                let a_zero_point = #a_zero_point.unsqueeze_dims(a_expansion_dim);

                let b_expansion_dim = if #b_scale.len() == #b.dims()[0] { 1 } else { 0 };
                let b_scale = #b_scale.unsqueeze_dims(b_expansion_dim);
                let b_zero_point = #b_zero_point.unsqueeze_dims(b_expansion_dim);
            }
        } else {
            // Case 3: Scale and zero_points have the same rank as their operands. Either the last or second-to-last dimension has a size of 1.
            // Reshaping is not required.
            quote! {}
        };

        if matches!(a_arg.ty.elem_type(), DType::QFloat(..))
            || matches!(b_arg.ty.elem_type(), DType::QFloat(..))
        {
            panic!("Quantized floats are not supported in `burn`") // Actually in the burn-onnx codegen
        }

        // Convert matmul operands and zero_points into floats
        // NOTE: A faster path can be achieved if both `a` and `b` are int types,
        //       by performing matmul in I32 before applying the scale inputs.
        //       See https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_qlinear_matmul.py
        let a_float = to_float(a_arg, a);
        let a_zero_point_float = to_float(a_zero_point_arg, a_zero_point);
        let b_float = to_float(b_arg, b);
        let b_zero_point_float = to_float(b_zero_point_arg, b_zero_point);

        quote! {
            #reshape_code

            let a_dequantized = #a_scale * (#a_float - #a_zero_point_float);
            let b_dequantized = #b_scale * (#b_float - #b_zero_point_float);

            let output_tensor = a_dequantized.matmul(b_dequantized);
            let #output = output_tensor.div(#y_scale).round().cast(#output_dtype) + #y_zero_point; // Quantized
        }
    }
}

/// Cast the tensor and zero points to float
fn to_float(arg: &Argument, token_stream: TokenStream) -> TokenStream {
    if arg.ty.elem_type().is_int() {
        quote! { #token_stream.cast(burn::tensor::DType::I32).float() }
    } else {
        quote! { #token_stream }
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
            .input_tensor("a_scale", 0, DType::F32)
            .input_tensor("a_zero_point", 0, DType::I8)
            .input_tensor("b", 2, DType::I8)
            .input_tensor("b_scale", 0, DType::F32)
            .input_tensor("b_zero_point", 0, DType::I8)
            .input_tensor("y_scale", 0, DType::F32)
            .input_tensor("y_zero_point", 0, DType::I8)
            .output_tensor("y", 2, DType::I8)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            a_scale: Tensor<B, 0>,
            a_zero_point: Tensor<B, 0, Int>,
            b: Tensor<B, 2, Int>,
            b_scale: Tensor<B, 0>,
            b_zero_point: Tensor<B, 0, Int>,
            y_scale: Tensor<B, 0>,
            y_zero_point: Tensor<B, 0, Int>,
        ) -> Tensor<B, 2, Int> {
            let a_dequantized = a_scale
                * (a.cast(burn::tensor::DType::I32).float()
                    - a_zero_point.cast(burn::tensor::DType::I32).float());
            let b_dequantized = b_scale
                * (b.cast(burn::tensor::DType::I32).float()
                    - b_zero_point.cast(burn::tensor::DType::I32).float());
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = output_tensor.div(y_scale).round().cast(burn::tensor::DType::I8)
                + y_zero_point;
            y
        }
        ");
    }

    #[test]
    fn test_qlinear_matmul_case_1_scalar_different_output_dtype() {
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor("a", 2, DType::I8)
            .input_tensor("a_scale", 0, DType::F32)
            .input_tensor("a_zero_point", 0, DType::I8)
            .input_tensor("b", 2, DType::I8)
            .input_tensor("b_scale", 0, DType::F32)
            .input_tensor("b_zero_point", 0, DType::I8)
            .input_tensor("y_scale", 0, DType::F32)
            .input_tensor("y_zero_point", 0, DType::BF16)
            .output_tensor("y", 2, DType::BF16) // Different from `a` and `b`
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            a_scale: Tensor<B, 0>,
            a_zero_point: Tensor<B, 0, Int>,
            b: Tensor<B, 2, Int>,
            b_scale: Tensor<B, 0>,
            b_zero_point: Tensor<B, 0, Int>,
            y_scale: Tensor<B, 0>,
            y_zero_point: Tensor<B, 0>,
        ) -> Tensor<B, 2> {
            let a_dequantized = a_scale
                * (a.cast(burn::tensor::DType::I32).float()
                    - a_zero_point.cast(burn::tensor::DType::I32).float());
            let b_dequantized = b_scale
                * (b.cast(burn::tensor::DType::I32).float()
                    - b_zero_point.cast(burn::tensor::DType::I32).float());
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = output_tensor.div(y_scale).round().cast(burn::tensor::DType::BF16)
                + y_zero_point;
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
            .input_tensor("a_scale", 0, DType::F32)
            .input_tensor(
                "a_zero_point",
                0,
                DType::QFloat(QuantScheme::default().with_value(QuantValue::E5M2)),
            )
            .input_tensor(
                "b",
                2,
                DType::QFloat(QuantScheme::default().with_value(QuantValue::E5M2)),
            )
            .input_tensor("b_scale", 0, DType::F32)
            .input_tensor(
                "b_zero_point",
                0,
                DType::QFloat(QuantScheme::default().with_value(QuantValue::E5M2)),
            )
            .input_tensor("y_scale", 0, DType::F32)
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
            .input_tensor("y_scale", 0, DType::F32)
            .input_tensor("y_zero_point", 0, DType::I8)
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
            y_scale: Tensor<B, 0>,
            y_zero_point: Tensor<B, 0, Int>,
        ) -> Tensor<B, 2, Int> {
            let a_expansion_dim = if a_scale.len() == a.dims()[0] { 1 } else { 0 };
            let a_scale = a_scale.unsqueeze_dims(a_expansion_dim);
            let a_zero_point = a_zero_point.unsqueeze_dims(a_expansion_dim);
            let b_expansion_dim = if b_scale.len() == b.dims()[0] { 1 } else { 0 };
            let b_scale = b_scale.unsqueeze_dims(b_expansion_dim);
            let b_zero_point = b_zero_point.unsqueeze_dims(b_expansion_dim);
            let a_dequantized = a_scale
                * (a.cast(burn::tensor::DType::I32).float()
                    - a_zero_point.cast(burn::tensor::DType::I32).float());
            let b_dequantized = b_scale
                * (b.cast(burn::tensor::DType::I32).float()
                    - b_zero_point.cast(burn::tensor::DType::I32).float());
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = output_tensor.div(y_scale).round().cast(burn::tensor::DType::I8)
                + y_zero_point;
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
            .input_tensor("y_scale", 0, DType::F32)
            .input_tensor("y_zero_point", 0, DType::I8)
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
            y_scale: Tensor<B, 0>,
            y_zero_point: Tensor<B, 0, Int>,
        ) -> Tensor<B, 3, Int> {
            let a_dequantized = a_scale
                * (a.cast(burn::tensor::DType::I32).float()
                    - a_zero_point.cast(burn::tensor::DType::I32).float());
            let b_dequantized = b_scale
                * (b.cast(burn::tensor::DType::I32).float()
                    - b_zero_point.cast(burn::tensor::DType::I32).float());
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = output_tensor.div(y_scale).round().cast(burn::tensor::DType::I8)
                + y_zero_point;
            y
        }
        ");
    }
}
