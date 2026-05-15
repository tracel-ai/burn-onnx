// Include the models for this node type
use crate::include_models;
include_models!(scaler, scaler_per_feature_3d, scaler_i64);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Int, Tensor, TensorData};

    #[test]
    fn test_scaler() {
        // Initialize the model
        let model: scaler::Model = scaler::Model::default();

        let device = Default::default();

        // Input: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        // Formula: Y = (X - offset) * scale
        // With scale=2.0 and offset=1.0: Y = (X - 1.0) * 2.0
        let input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

        let output = model.forward(input);

        // Expected: [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]
        let expected = TensorData::from([[0.0f32, 2.0, 4.0], [6.0, 8.0, 10.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_scaler_per_feature_3d() {
        // Rank-3 F32 input with per-feature scale/offset.
        // Exercises the [1, ..., 1, F] reshape-to-last-axis broadcast path.
        // Model: scale=[1.0, 2.0, 0.5], offset=[0.0, 1.0, 2.0]
        // Formula per last-axis feature: Y[..., i] = (X[..., i] - offset[i]) * scale[i]
        let model: scaler_per_feature_3d::Model = scaler_per_feature_3d::Model::default();

        let device = Default::default();

        let input = Tensor::<3>::from_floats(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            &device,
        );

        let output = model.forward(input);

        // Expected:
        // [[[1-0)*1, (2-1)*2, (3-2)*0.5], [(4-0)*1, (5-1)*2, (6-2)*0.5]],
        //  [[(7-0)*1, (8-1)*2, (9-2)*0.5], [(10-0)*1, (11-1)*2, (12-2)*0.5]]]
        // = [[[1, 2, 0.5], [4, 8, 2]], [[7, 14, 3.5], [10, 20, 5]]]
        let expected = TensorData::from([
            [[1.0f32, 2.0, 0.5], [4.0, 8.0, 2.0]],
            [[7.0, 14.0, 3.5], [10.0, 20.0, 5.0]],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_scaler_i64() {
        // I64 integer input with per-feature scale/offset.
        // Exercises the .int().cast(DType::F32) codegen path.
        // Model: scale=[3.0, 2.0, 1.0], offset=[2.0, 2.0, 2.0]
        let model: scaler_i64::Model = scaler_i64::Model::default();

        let device = Default::default();

        let input = Tensor::<2, Int>::from_ints([[2, 4, 6], [8, 10, 12]], &device);

        let output = model.forward(input);

        // Expected: [(2-2)*3, (4-2)*2, (6-2)*1] = [0, 4, 4]
        //           [(8-2)*3, (10-2)*2, (12-2)*1] = [18, 16, 10]
        let expected = TensorData::from([[0.0f32, 4.0, 4.0], [18.0, 16.0, 10.0]]);

        output.to_data().assert_eq(&expected, true);
    }
}
