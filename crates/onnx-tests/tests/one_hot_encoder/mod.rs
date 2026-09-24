use crate::include_models;
include_models!(
    one_hot_encoder_f32,
    one_hot_encoder_f32_large_cats,
    one_hot_encoder_f64,
    one_hot_encoder_i64,
    one_hot_encoder_2d
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    #[test]
    fn one_hot_encoder_f32_input() {
        let device = Default::default();
        let model = one_hot_encoder_f32::Model::new(&device);

        let input: Tensor<1> = Tensor::from_data(
            TensorData::from([1.0f32, 4.0, 2.0, 1.0]),
            (&device, DType::F32),
        );
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([
            [1.0f32, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_encoder_f32_large_categories() {
        // Categories above 2^24 are not exact in f32. The spec casts float input
        // to integers, so 16777216.0 matches 16777216 but not 16777217.
        let device = Default::default();
        let model = one_hot_encoder_f32_large_cats::Model::new(&device);

        let input: Tensor<1> = Tensor::from_data(
            TensorData::from([16777216.0f32, 5.0]),
            (&device, DType::F32),
        );
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([[1.0f32, 0.0], [0.0, 0.0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    #[cfg_attr(feature = "test-metal", ignore = "Metal has no f64")]
    fn one_hot_encoder_f64_input() {
        let device = burn::tensor::Device::default();
        // f64 support on wgpu depends on the adapter.
        if !device.supports_dtype(DType::F64) {
            return;
        }
        let model = one_hot_encoder_f64::Model::new(&device);

        let input: Tensor<1> = Tensor::from_data(
            TensorData::from([4.0f64, 2.0, 3.0, 1.0]),
            (&device, DType::F64),
        );
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([
            [0.0f32, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_encoder_i64_input() {
        let device = Default::default();
        let model = one_hot_encoder_i64::Model::new(&device);

        // cats_int64s = [10, 20, 30], so the category value is not its column
        // index. 7 is out of vocabulary and yields an all-zero row.
        let input = Tensor::<1, Int>::from_data(
            TensorData::from([30i64, 10, 7, 20]),
            (&device, DType::I64),
        );
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([
            [0.0f32, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_encoder_2d_input() {
        let device = Default::default();
        let model = one_hot_encoder_2d::Model::new(&device);

        // cats_int64s = [1, 2, 4]; 9 is out of vocabulary.
        let input = Tensor::<2, Int>::from_data(
            TensorData::from([[1i32, 4], [2, 9]]),
            (&device, DType::I32),
        );
        let output: Tensor<3> = model.forward(input);

        let expected = TensorData::from([
            [[1.0f32, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ]);
        output.to_data().assert_eq(&expected, true);
    }
}
