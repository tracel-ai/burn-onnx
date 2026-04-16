use crate::include_models;
include_models!(
    qlinear_matmul_scalar,
    qlinear_matmul_vector,
    qlinear_matmul_nd,
    qlinear_matmul_u8_saturate,
    qlinear_matmul_i8_saturate,
    qlinear_matmul_opset_10,
    qlinear_matmul_scalar_f16_scale,
    qlinear_matmul_vector_bf16_scale
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData};
    use half::{bf16, f16};

    use crate::backend::TestBackend;

    // Case 1: 3D operands with scalar (per-tensor) quantization parameters.
    #[test]
    fn qlinear_matmul_scalar() {
        let device = Default::default();
        let model: qlinear_matmul_scalar::Model<TestBackend> =
            qlinear_matmul_scalar::Model::new(&device);

        let a = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::from([
                [[6u8, 1, 19, 10], [11, 3, 2, 19]],
                [[14, 14, 10, 3], [7, 7, 14, 1]],
            ]),
            (&device, DType::U8),
        );
        let b = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::from([
                [[12u8, 11, 6], [16, 7, 2], [18, 18, 1], [15, 7, 10]],
                [[17, 14, 10], [17, 13, 13], [3, 12, 2], [7, 11, 18]],
            ]),
            (&device, DType::U8),
        );

        let output = model.forward(a, 0.1f32, 2u8, b, 0.2f32, 3u8, 0.3f32, 4u8);

        let expected =
            TensorData::from([[[29u8, 25, 6], [24, 14, 14]], [[27, 26, 18], [13, 18, 8]]]);
        output.to_data().assert_eq(&expected, true);
    }

    // Case 2: 2D operands with rank-1 (per-axis) quantization parameters.
    // a uses per-row scales (shape [M]), b uses per-column scales (shape [N]),
    // y uses per-row scales (shape [M]).
    #[test]
    fn qlinear_matmul_vector() {
        let device = Default::default();
        let model: qlinear_matmul_vector::Model<TestBackend> =
            qlinear_matmul_vector::Model::new(&device);

        let a = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[6u8, 1, 19, 10], [11, 3, 2, 19]]),
            (&device, DType::U8),
        );
        let a_scale = Tensor::<TestBackend, 1>::from_floats([0.19160044, 0.7818941], &device);
        let a_zero_point = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([4u8, 1]),
            (&device, DType::U8),
        );

        let b = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[18u8, 1, 15], [7, 10, 17], [14, 10, 17], [13, 13, 3]]),
            (&device, DType::U8),
        );
        let b_scale =
            Tensor::<TestBackend, 1>::from_floats([0.15143815, 0.6543796, 0.06584746], &device);
        let b_zero_point = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([3u8, 1, 3]),
            (&device, DType::U8),
        );

        let y_scale = Tensor::<TestBackend, 1>::from_floats([1.0f32, 0.5], &device);
        let y_zero_point = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([10u8, 5]),
            (&device, DType::U8),
        );

        let output = model.forward(
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
        );

        let expected = TensorData::from([[17u8, 33, 12], [88, 254, 22]]);
        output.to_data().assert_eq(&expected, true);
    }

    // Case 3: 3D operands with N-D quantization parameters.
    // a_scale: [batch, M, 1] (per-row per-batch), b_scale: [batch, 1, N] (per-column per-batch),
    // y_scale: [batch, M, 1] (per-row per-batch).
    #[test]
    fn qlinear_matmul_nd() {
        let device = Default::default();
        let model: qlinear_matmul_nd::Model<TestBackend> = qlinear_matmul_nd::Model::new(&device);

        let a = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::from([
                [[6u8, 1, 19, 10], [11, 3, 2, 19]],
                [[14, 14, 10, 3], [7, 7, 14, 1]],
            ]),
            (&device, DType::U8),
        );
        let a_scale = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[0.60088164f32], [0.4513744]], [[0.10897516], [0.4646564]]]),
            &device,
        );
        let a_zero_point = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::from([[[4u8], [3]], [[2], [4]]]),
            (&device, DType::U8),
        );

        let b = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::from([
                [[1u8, 3, 12], [9, 11, 4], [16, 5, 1], [9, 19, 0]],
                [[12, 14, 19], [1, 19, 14], [0, 4, 19], [11, 17, 2]],
            ]),
            (&device, DType::U8),
        );
        let b_scale = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([
                [[0.61553663f32, 0.01699564, 0.0328318]],
                [[0.5295269, 0.40586236, 0.05619901]],
            ]),
            &device,
        );
        let b_zero_point = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::from([[[3u8, 2, 3]], [[3, 0, 2]]]),
            (&device, DType::U8),
        );

        let y_scale = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[1.7899106f32], [1.5204613]], [[1.1757488], [0.51989746]]]),
            &device,
        );
        let y_zero_point = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::from([[[8u8], [11]], [[13], [1]]]),
            (&device, DType::U8),
        );

        let output = model.forward(
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
        );

        let expected =
            TensorData::from([[[51u8, 9, 8], [23, 12, 11]], [[16, 30, 16], [0, 33, 14]]]);
        output.to_data().assert_eq(&expected, true);
    }

    // Case 4: U8 saturation test — verifies upper (255) and lower (0) U8 clamp fires correctly.
    // Inputs are engineered so that two output positions overflow the U8 range before clamping:
    //   [0,0]: round(4×118×117×0.02/0.3)+4 = 3686 → 255  (upper saturation)
    //   [0,1]: round(4×118×2×0.02/0.3)+4   =   67        (in range)
    //   [1,0]: round(4×-2×117×0.02/0.3)+4  =  -58 →   0  (lower saturation)
    //   [1,1]: round(4×-2×2×0.02/0.3)+4    =    3        (in range)
    // Expected values are hand-computed with explicit clip — NOT from the ONNX ReferenceEvaluator,
    // which wraps on overflow rather than saturating (see https://github.com/onnx/onnx/issues/7835).
    #[test]
    fn qlinear_matmul_u8_saturate() {
        let device = Default::default();
        let model: qlinear_matmul_u8_saturate::Model<TestBackend> =
            qlinear_matmul_u8_saturate::Model::new(&device);

        let a = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[120u8, 120, 120, 120], [0, 0, 0, 0]]),
            (&device, DType::U8),
        );
        let b = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[120u8, 5], [120, 5], [120, 5], [120, 5]]),
            (&device, DType::U8),
        );

        let output = model.forward(a, 0.1f32, 2u8, b, 0.2f32, 3u8, 0.3f32, 4u8);

        let expected = TensorData::from([[255u8, 67], [0, 3]]);
        output.to_data().assert_eq(&expected, true);
    }

    // Case 5: I8 saturation test — exercises clamp(-128, 127) and the I8 cast.
    // Uses symmetric quantization (zero_point=0) with large positive and large negative values.
    //   [0,0]: round(4×100×100×0.1×0.2/0.1)+0 = 8000 → 127   (upper saturation)
    //   [0,1]: round(4×100×1×0.1×0.2/0.1)+0   =   80          (no saturation)
    //   [1,0]: round(4×-100×100×0.1×0.2/0.1)+0 = -8000 → -128 (lower saturation)
    //   [1,1]: round(4×-100×1×0.1×0.2/0.1)+0   =  -80          (no saturation)
    // Expected values are hand-computed with explicit clip — NOT from the ONNX ReferenceEvaluator,
    // which wraps on overflow rather than saturating (see https://github.com/onnx/onnx/issues/7835).
    #[test]
    fn qlinear_matmul_i8_saturate() {
        let device = Default::default();
        let model: qlinear_matmul_i8_saturate::Model<TestBackend> =
            qlinear_matmul_i8_saturate::Model::new(&device);

        let a = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[100i8, 100, 100, 100], [-100, -100, -100, -100]]),
            (&device, DType::I8),
        );
        let b = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[100i8, 1], [100, 1], [100, 1], [100, 1]]),
            (&device, DType::I8),
        );

        let output = model.forward(a, 0.1f32, 0, b, 0.2f32, 0, 0.1f32, 0);

        let expected = TensorData::from([[127i8, 80], [-128, -80]]);
        output.to_data().assert_eq(&expected, true);
    }

    // Case 6: opset-10 model (2D operands, U8 scalar, F16 scales)
    #[test]
    fn qlinear_matmul_opset_10() {
        let device = Default::default();
        let model: qlinear_matmul_opset_10::Model<TestBackend> =
            qlinear_matmul_opset_10::Model::new(&device);

        let a = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[6u8, 1, 19, 10], [11, 3, 2, 19]]),
            (&device, DType::U8),
        );
        let b = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[14u8, 14, 10], [3, 7, 7], [14, 1, 12], [11, 6, 16]]),
            (&device, DType::U8),
        );

        let output = model.forward(
            a,
            0.0999755859375f32,
            2u8,
            b,
            0.199951171875f32,
            3u8,
            0.300048828125f32,
            4u8,
        );

        let expected = TensorData::from([[24u8, 6, 23], [20, 14, 23]]);
        output.to_data().assert_eq(&expected, true);
    }

    // Case 7: scalar F16 scales — verifies the `(scale as f32)` cast path for half-precision scalars.
    // Same operand values as Case 6; scales are the nearest F16 representations of 0.1/0.2/0.3.
    #[test]
    fn qlinear_matmul_scalar_f16_scale() {
        let device = Default::default();
        let model: qlinear_matmul_scalar_f16_scale::Model<TestBackend> =
            qlinear_matmul_scalar_f16_scale::Model::new(&device);

        let a = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[6u8, 1, 19, 10], [11, 3, 2, 19]]),
            (&device, DType::U8),
        );
        let b = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[14u8, 14, 10], [3, 7, 7], [14, 1, 12], [11, 6, 16]]),
            (&device, DType::U8),
        );

        let output = model.forward(
            a,
            f16::from_f32(0.1),
            2u8,
            b,
            f16::from_f32(0.2),
            3u8,
            f16::from_f32(0.3),
            4u8,
        );

        let expected = TensorData::from([[24u8, 6, 23], [20, 14, 23]]);
        output.to_data().assert_eq(&expected, true);
    }

    // Case 8: vector BF16 scales — verifies the `.cast(DType::F32)` path for half-precision tensors.
    // Same operand values as Case 2; scales are the nearest BF16 representations of the F32 values.
    #[test]
    fn qlinear_matmul_vector_bf16_scale() {
        let device = Default::default();
        let model: qlinear_matmul_vector_bf16_scale::Model<TestBackend> =
            qlinear_matmul_vector_bf16_scale::Model::new(&device);

        let a = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[6u8, 1, 19, 10], [11, 3, 2, 19]]),
            (&device, DType::U8),
        );
        let a_scale = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([bf16::from_f32(0.19160044), bf16::from_f32(0.7818941)]),
            &device,
        );
        let a_zero_point = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([4u8, 1]),
            (&device, DType::U8),
        );

        let b = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[18u8, 1, 15], [7, 10, 17], [14, 10, 17], [13, 13, 3]]),
            (&device, DType::U8),
        );
        let b_scale = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([
                bf16::from_f32(0.15143815),
                bf16::from_f32(0.6543796),
                bf16::from_f32(0.06584746),
            ]),
            &device,
        );
        let b_zero_point = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([3u8, 1, 3]),
            (&device, DType::U8),
        );

        let y_scale = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([bf16::from_f32(1.0), bf16::from_f32(0.5)]),
            &device,
        );
        let y_zero_point = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([10u8, 5]),
            (&device, DType::U8),
        );

        let output = model.forward(
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
        );

        // NOTE: The test expects [88, 254, 22] at row 1, not [87, ...] from Python's ReferenceEvaluator.
        // The ReferenceEvaluator computes in BF16 throughout; Rust casts BF16→F32 then computes in
        // F32. For [1,0]: BF16 intermediate product rounds to 0.11816 (→ 87), F32 gives 0.11825 (→ 88).
        let expected = TensorData::from([[17u8, 33, 12], [88, 254, 22]]);
        output.to_data().assert_eq(&expected, true);
    }
}
