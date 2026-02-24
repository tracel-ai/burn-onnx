use crate::include_models;
include_models!(quantize_linear, quantize_linear_axis);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn quantize_linear() {
        let device = Default::default();
        let model: quantize_linear::Model<TestBackend> = quantize_linear::Model::new(&device);

        let input = Tensor::<TestBackend, 2>::from_floats([[-1.0, 0.0, 1.0, 2.1]], &device);
        let scale = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let zero_data = TensorData::from([3i16]);
        let zero_point =
            Tensor::<TestBackend, 1, Int>::from_data_dtype(zero_data, &device, DType::I16);

        let output = model.forward(input, scale, zero_point);

        let expected = TensorData::from([[1i16, 3, 5, 7]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn quantize_linear_per_axis() {
        let device = Default::default();
        let model: quantize_linear_axis::Model<TestBackend> =
            quantize_linear_axis::Model::new(&device);

        let input =
            Tensor::<TestBackend, 2>::from_floats([[-1.0, 0.0, 1.0], [2.0, 4.0, 6.0]], &device);
        let scale = Tensor::<TestBackend, 1>::from_floats([0.5, 2.0], &device);
        let zero_data = TensorData::from([3i16, 10i16]);
        let zero_point =
            Tensor::<TestBackend, 1, Int>::from_data_dtype(zero_data, &device, DType::I16);

        let output = model.forward(input, scale, zero_point);

        let expected = TensorData::from([[1i16, 3, 5], [11, 12, 13]]);
        output.to_data().assert_eq(&expected, true);
    }
}
