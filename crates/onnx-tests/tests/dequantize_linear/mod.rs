use crate::include_models;
include_models!(dequantize_linear, dequantize_linear_axis);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn dequantize_linear() {
        let device = Default::default();
        let model: dequantize_linear::Model<TestBackend> = dequantize_linear::Model::new(&device);

        let input_data = TensorData::from([[2i32, 4, 6, 10]]);
        let input = Tensor::<TestBackend, 2, Int>::from_data_dtype(input_data, &device, DType::I32);
        let scale = Tensor::<TestBackend, 1>::from_floats([0.5], &device);

        let output = model.forward(input, scale);

        let expected = TensorData::from([[1.0f32, 2.0, 3.0, 5.0]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(1e-5, 1e-6));
    }

    #[test]
    fn dequantize_linear_per_axis() {
        let device = Default::default();
        let model: dequantize_linear_axis::Model<TestBackend> =
            dequantize_linear_axis::Model::new(&device);

        let input_data = TensorData::from([[2i32, 4, 6], [1, 3, 5]]);
        let input = Tensor::<TestBackend, 2, Int>::from_data_dtype(input_data, &device, DType::I32);
        let scale = Tensor::<TestBackend, 1>::from_floats([0.5, 2.0], &device);

        let output = model.forward(input, scale);

        let expected = TensorData::from([[1.0f32, 2.0, 3.0], [2.0, 6.0, 10.0]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(1e-5, 1e-6));
    }
}
