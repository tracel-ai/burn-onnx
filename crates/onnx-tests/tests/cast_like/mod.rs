// Import the shared macro
use crate::include_models;
include_models!(cast_like);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Int, Tensor, TensorData, Tolerance};

    #[test]
    fn cast_like() {
        let device = Default::default();
        let model: cast_like::Model = cast_like::Model::new(&device);

        // float_input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] shape [2, 3]
        let float_input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        // int_target: same shape, used only for dtype (i64)
        let int_target = Tensor::<2, Int>::from_ints([[0, 0, 0], [0, 0, 0]], &device);
        // int_input: [1, 2, 3, 4, 5, 6] shape [2, 3]
        let int_input = Tensor::<2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device);
        // float_target: same shape, used only for dtype (f32)
        let float_target = Tensor::<2>::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);

        let (float_to_int, int_to_float) = model.forward(
            float_input.clone(),
            int_target,
            int_input.clone(),
            float_target,
        );

        // float -> int: values should be truncated integers
        let expected_int = TensorData::from([[1i64, 2i64, 3i64], [4i64, 5i64, 6i64]]);
        float_to_int.to_data().assert_eq(&expected_int, true);

        // int -> float: values should be the same as int_input cast to float
        let expected_float = TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        int_to_float
            .to_data()
            .assert_approx_eq::<f32>(&expected_float, Tolerance::default());
    }
}
