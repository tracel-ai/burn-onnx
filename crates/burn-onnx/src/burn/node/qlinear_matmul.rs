use crate::burn::node::matmul::matmul_forward;

use super::prelude::*;

impl NodeCodegen for onnx_ir::qlinear_matmul::QLinearMatMulNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // Assumptions (per the ONNX spec):
        // Scale and zero point input for a given operand have the same shape.

        // QFloat operands are rejected upstream (see onnx_ir::qlinear_matmul),
        // so here both operand tensors and their zero points are guaranteed to be I8 or U8.

        let a_arg = self.a();
        let a_scale_arg = self.a_scale();
        let a_zero_point_arg = self.a_zero_point();
        let b_arg = self.b();
        let b_scale_arg = self.b_scale();
        let b_zero_point_arg = self.b_zero_point();
        let y_scale_arg = self.y_scale();
        let y_zero_point_arg = self.y_zero_point();

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

        let reshape_a_scale_and_zp = reshape_scale_and_zp(a_scale_arg, &a, &a_scale, &a_zero_point);
        let reshape_b_scale_and_zp = reshape_scale_and_zp(b_scale_arg, &b, &b_scale, &b_zero_point);

        // Generate y reshape code by passing a TokenStream for `output_tensor`.
        // This will be inserted after `output_tensor` is created in the quote block.
        let output_tensor_ts = quote! { output_tensor };
        let reshape_y_scale_and_zp =
            reshape_scale_and_zp(y_scale_arg, &output_tensor_ts, &y_scale, &y_zero_point);

        // Convert inputs into floats
        // NOTE: A faster path can be achieved if both `a`, `b` and their zero points are int dtypes,
        //       by performing matmul in I32 before applying the scale inputs.
        //       See https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_qlinear_matmul.py
        let a_float = to_float(a_arg, a);
        let a_zero_point_float = to_float(a_zero_point_arg, a_zero_point);
        let a_scale = to_float(a_scale_arg, a_scale);
        let b_float = to_float(b_arg, b);
        let b_scale = to_float(b_scale_arg, b_scale);
        let b_zero_point_float = to_float(b_zero_point_arg, b_zero_point);
        let y_zero_point_float = to_float(y_zero_point_arg, y_zero_point);
        let y_scale = to_float(y_scale_arg, y_scale);

        let matmul_statement = matmul_forward(
            quote! { a_dequantized },
            quote! { b_dequantized },
            output_tensor_ts,
            a_arg.ty.rank(),
            b_arg.ty.rank(),
            self.outputs[0].ty.rank(),
        );

        quote! {
            // Dequantize inputs
            #reshape_a_scale_and_zp
            #reshape_b_scale_and_zp
            let a_dequantized = #a_scale * (#a_float - #a_zero_point_float);
            let b_dequantized = #b_scale * (#b_float - #b_zero_point_float);

            #matmul_statement

            // Quantize output
            #reshape_y_scale_and_zp
            let #output = (output_tensor / #y_scale).round();
            let #output = (#output + #y_zero_point_float)#clamp_expr.int().cast(#output_dtype);
        }
    }
}

/// Cast a variable token to float
fn to_float(arg: &Argument, token_stream: TokenStream) -> TokenStream {
    let dtype = arg.ty.elem_type();
    if dtype.is_int() || dtype.is_uint() {
        if arg.ty.is_scalar() {
            quote! { f32::from(#token_stream) }
        } else {
            let f32_dtype = DType::F32.to_tokens();
            quote! { #token_stream.float().cast(#f32_dtype) }
        }
    } else if dtype == DType::F32 {
        token_stream
    } else if matches!(dtype, DType::F16 | DType::BF16) {
        if arg.ty.is_scalar() {
            quote! { f32::from(#token_stream) }
        } else {
            let f32_dtype = DType::F32.to_tokens();
            quote! {
                #token_stream.cast(#f32_dtype)
            }
        }
    } else {
        unreachable!("Unsupported type {dtype:?}")
    }
}

fn reshape_scale_and_zp(
    scale_arg: &Argument,
    tensor: &TokenStream,
    scale: &TokenStream,
    zero_point: &TokenStream,
) -> TokenStream {
    // Reshape the scale and zero point, if necessary.
    // There are three possible quantization cases, based on the scale and zero point shapes:
    //   1. Scalar; per-tensor quantization
    //   2. Vectors; row/column-based quantization
    //   3. N-D tensors; higher-dim row/column-based quantization
    if scale_arg.ty.is_scalar() {
        // Case 1: All zero point and inputs are scalars
        // Reshaping is not required.
        quote! {}
    } else if scale_arg.ty.rank() == 1 {
        // Case 2: Scale and zero point are vectors. Infer whether they are row or column vectors and expand appropriate.
        quote! {
            let (#scale, #zero_point) = {
                let expansion_dim = if #scale.dims()[0] == #tensor.dims()[0] { 1 } else { 0 };
                (#scale.unsqueeze_dim(expansion_dim), #zero_point.unsqueeze_dim(expansion_dim))
            };
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
    use burn::tensor::DType;
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
                * (a.float().cast(burn::tensor::DType::F32) - f32::from(a_zero_point));
            let b_dequantized = b_scale
                * (b.float().cast(burn::tensor::DType::F32) - f32::from(b_zero_point));
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = (output_tensor / y_scale).round();
            let y = (y + f32::from(y_zero_point))
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
            .input_scalar("y_zero_point", DType::U8)
            .output_tensor("y", 2, DType::U8) // Different from `a` and `b`
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
            y_zero_point: u8,
        ) -> Tensor<B, 2, Int> {
            let a_dequantized = a_scale
                * (a.float().cast(burn::tensor::DType::F32) - f32::from(a_zero_point));
            let b_dequantized = b_scale
                * (b.float().cast(burn::tensor::DType::F32) - f32::from(b_zero_point));
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = (output_tensor / y_scale).round();
            let y = (y + f32::from(y_zero_point))
                .clamp(0f32, 255f32)
                .int()
                .cast(burn::tensor::DType::U8);
            y
        }
        ");
    }

    #[test]
    fn test_qlinear_matmul_case_1_scalar_different_scale_dtype() {
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor("a", 2, DType::I8)
            .input_scalar("a_scale", DType::F16)
            .input_scalar("a_zero_point", DType::I8)
            .input_tensor("b", 2, DType::I8)
            .input_scalar("b_scale", DType::F16)
            .input_scalar("b_zero_point", DType::I8)
            .input_scalar("y_scale", DType::BF16)
            .input_scalar("y_zero_point", DType::U8)
            .output_tensor("y", 2, DType::U8)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            a_scale: half::f16,
            a_zero_point: i8,
            b: Tensor<B, 2, Int>,
            b_scale: half::f16,
            b_zero_point: i8,
            y_scale: half::bf16,
            y_zero_point: u8,
        ) -> Tensor<B, 2, Int> {
            let a_dequantized = f32::from(a_scale)
                * (a.float().cast(burn::tensor::DType::F32) - f32::from(a_zero_point));
            let b_dequantized = f32::from(b_scale)
                * (b.float().cast(burn::tensor::DType::F32) - f32::from(b_zero_point));
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = (output_tensor / f32::from(y_scale)).round();
            let y = (y + f32::from(y_zero_point))
                .clamp(0f32, 255f32)
                .int()
                .cast(burn::tensor::DType::U8);
            y
        }
        ");
    }

    #[test]
    fn test_qlinear_matmul_case_1_matrix_matmul_vector() {
        // 2D matrix × 1D vector → 1D output (matmul_output_rank(2, 1) = 1)
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor("a", 2, DType::I8) // Matrix operand
            .input_scalar("a_scale", DType::F32)
            .input_scalar("a_zero_point", DType::I8)
            .input_tensor("b", 1, DType::I8) // Vector operand
            .input_scalar("b_scale", DType::F32)
            .input_scalar("b_zero_point", DType::I8)
            .input_scalar("y_scale", DType::F32)
            .input_scalar("y_zero_point", DType::U8)
            .output_tensor("y", 1, DType::U8) // Output is rank 1: [M, K] @ [K] → [M]
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            a_scale: f32,
            a_zero_point: i8,
            b: Tensor<B, 1, Int>,
            b_scale: f32,
            b_zero_point: i8,
            y_scale: f32,
            y_zero_point: u8,
        ) -> Tensor<B, 1, Int> {
            let a_dequantized = a_scale
                * (a.float().cast(burn::tensor::DType::F32) - f32::from(a_zero_point));
            let b_dequantized = b_scale
                * (b.float().cast(burn::tensor::DType::F32) - f32::from(b_zero_point));
            let output_tensor = a_dequantized
                .matmul(b_dequantized.unsqueeze_dims(&[-1isize]))
                .squeeze_dim::<1usize>(1usize);
            let y = (output_tensor / y_scale).round();
            let y = (y + f32::from(y_zero_point))
                .clamp(0f32, 255f32)
                .int()
                .cast(burn::tensor::DType::U8);
            y
        }
        ");
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
            let (a_scale, a_zero_point) = {
                let expansion_dim = if a_scale.dims()[0] == a.dims()[0] { 1 } else { 0 };
                (a_scale.unsqueeze_dim(expansion_dim), a_zero_point.unsqueeze_dim(expansion_dim))
            };
            let (b_scale, b_zero_point) = {
                let expansion_dim = if b_scale.dims()[0] == b.dims()[0] { 1 } else { 0 };
                (b_scale.unsqueeze_dim(expansion_dim), b_zero_point.unsqueeze_dim(expansion_dim))
            };
            let a_dequantized = a_scale
                * (a.float().cast(burn::tensor::DType::F32)
                    - a_zero_point.float().cast(burn::tensor::DType::F32));
            let b_dequantized = b_scale
                * (b.float().cast(burn::tensor::DType::F32)
                    - b_zero_point.float().cast(burn::tensor::DType::F32));
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let (y_scale, y_zero_point) = {
                let expansion_dim = if y_scale.dims()[0] == output_tensor.dims()[0] {
                    1
                } else {
                    0
                };
                (y_scale.unsqueeze_dim(expansion_dim), y_zero_point.unsqueeze_dim(expansion_dim))
            };
            let y = (output_tensor / y_scale).round();
            let y = (y + y_zero_point.float().cast(burn::tensor::DType::F32))
                .clamp(-128f32, 127f32)
                .int()
                .cast(burn::tensor::DType::I8);
            y
        }
        ");
    }

    #[test]
    fn test_qlinear_matmul_case_2_vector_different_scale_dtype() {
        // Case 2: Vector scale/zero_points (rank 1) with 2D tensor operands
        // Vectors are expanded to match operand rank
        let node = QLinearMatMulNodeBuilder::new("qmm")
            .input_tensor("a", 2, DType::I8)
            .input_tensor("a_scale", 1, DType::BF16)
            .input_tensor("a_zero_point", 1, DType::I8)
            .input_tensor("b", 2, DType::I8)
            .input_tensor("b_scale", 1, DType::BF16)
            .input_tensor("b_zero_point", 1, DType::I8)
            .input_tensor("y_scale", 1, DType::F16)
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
            let (a_scale, a_zero_point) = {
                let expansion_dim = if a_scale.dims()[0] == a.dims()[0] { 1 } else { 0 };
                (a_scale.unsqueeze_dim(expansion_dim), a_zero_point.unsqueeze_dim(expansion_dim))
            };
            let (b_scale, b_zero_point) = {
                let expansion_dim = if b_scale.dims()[0] == b.dims()[0] { 1 } else { 0 };
                (b_scale.unsqueeze_dim(expansion_dim), b_zero_point.unsqueeze_dim(expansion_dim))
            };
            let a_dequantized = a_scale.cast(burn::tensor::DType::F32)
                * (a.float().cast(burn::tensor::DType::F32)
                    - a_zero_point.float().cast(burn::tensor::DType::F32));
            let b_dequantized = b_scale.cast(burn::tensor::DType::F32)
                * (b.float().cast(burn::tensor::DType::F32)
                    - b_zero_point.float().cast(burn::tensor::DType::F32));
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let (y_scale, y_zero_point) = {
                let expansion_dim = if y_scale.dims()[0] == output_tensor.dims()[0] {
                    1
                } else {
                    0
                };
                (y_scale.unsqueeze_dim(expansion_dim), y_zero_point.unsqueeze_dim(expansion_dim))
            };
            let y = (output_tensor / y_scale.cast(burn::tensor::DType::F32)).round();
            let y = (y + y_zero_point.float().cast(burn::tensor::DType::F32))
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
                * (a.float().cast(burn::tensor::DType::F32)
                    - a_zero_point.float().cast(burn::tensor::DType::F32));
            let b_dequantized = b_scale
                * (b.float().cast(burn::tensor::DType::F32)
                    - b_zero_point.float().cast(burn::tensor::DType::F32));
            let output_tensor = a_dequantized.matmul(b_dequantized);
            let y = (output_tensor / y_scale).round();
            let y = (y + y_zero_point.float().cast(burn::tensor::DType::F32))
                .clamp(-128f32, 127f32)
                .int()
                .cast(burn::tensor::DType::I8);
            y
        }
        ");
    }
}
