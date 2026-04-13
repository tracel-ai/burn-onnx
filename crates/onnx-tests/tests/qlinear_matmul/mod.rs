use crate::include_models;
include_models!(
    qlinear_matmul_scalar,
    qlinear_matmul_vector,
    qlinear_matmul_nd,
    qlinear_matmul_saturate
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData};

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

    // Case 4: Saturation test — verifies upper (255) and lower (0) U8 clamp fires correctly.
    // Inputs are engineered so that two output positions overflow the U8 range before clamping:
    //   [0,0]: round(4×118×117×0.02/0.3)+4 = 3686 → 255  (upper saturation)
    //   [0,1]: round(4×118×2×0.02/0.3)+4   =   67        (in range)
    //   [1,0]: round(4×-2×117×0.02/0.3)+4  =  -58 →   0  (lower saturation)
    //   [1,1]: round(4×-2×2×0.02/0.3)+4    =    3        (in range)
    // Expected values are hand-computed with explicit clip — NOT from the ONNX ReferenceEvaluator,
    // which wraps on overflow rather than saturating.
    #[test]
    fn qlinear_matmul_saturate() {
        let device = Default::default();
        let model: qlinear_matmul_saturate::Model<TestBackend> =
            qlinear_matmul_saturate::Model::new(&device);

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
}
