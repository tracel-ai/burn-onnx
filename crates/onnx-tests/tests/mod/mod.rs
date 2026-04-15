// Import the shared macro
use crate::include_models;
include_models!(
    modulo,
    mod_scalar,
    mod_remainder,
    mod_fmod,
    mod_broadcast_fixed,
    mod_broadcast_remainder_fixed,
    mod_shape
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn mod_tensor_by_tensor() {
        // Initialize the model
        let device = Default::default();
        let model: modulo::Model<TestBackend> = modulo::Model::new(&device);

        // Run the model
        let input_x = Tensor::<TestBackend, 3>::from_floats([[[5.3, -5.3, 7.5, -7.5]]], &device);
        let input_y = Tensor::<TestBackend, 3>::from_floats([[[2.0, 2.0, 3.0, 3.0]]], &device);
        let output = model.forward(input_x, input_y);

        // Expected output: fmod(x, y) for each element
        // Using the actual computed values from Python
        let expected = TensorData::from([[[1.3000002f32, -1.3000002, 1.5, -1.5]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mod_tensor_by_scalar() {
        // Initialize the model
        let device = Default::default();
        let model: mod_scalar::Model<TestBackend> = mod_scalar::Model::new(&device);

        // Run the model
        let input_x = Tensor::<TestBackend, 4>::from_floats([[[[5.3, -5.3, 7.5, -7.5]]]], &device);
        let scalar = 2.0f64;
        let output = model.forward(input_x, scalar);

        // Expected output: fmod(x, 2.0) for each element
        // Using the actual computed values from Python
        let expected = TensorData::from([[[[1.3000002f32, -1.3000002, 1.5, -1.5]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mod_remainder() {
        // Test fmod=0 (Python-style remainder)
        let device = Default::default();
        let model: mod_remainder::Model<TestBackend> = mod_remainder::Model::new(&device);

        let input_x = Tensor::<TestBackend, 3>::from_floats([[[5.3, -5.3, 7.5, -7.5]]], &device);
        let input_y = Tensor::<TestBackend, 3>::from_floats([[[2.0, 2.0, 3.0, 3.0]]], &device);
        let output = model.forward(input_x, input_y);

        // Expected: Python-style remainder where sign follows divisor
        // remainder(5.3, 2.0) = 1.3, remainder(-5.3, 2.0) = 0.7
        // remainder(7.5, 3.0) = 1.5, remainder(-7.5, 3.0) = 1.5
        let expected = TensorData::from([[[1.3000002f32, 0.6999998, 1.5, 1.5]]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mod_fmod() {
        // Test fmod=1 (C-style fmod)
        let device = Default::default();
        let model: mod_fmod::Model<TestBackend> = mod_fmod::Model::new(&device);

        let input_x = Tensor::<TestBackend, 3>::from_floats([[[5.3, -5.3, 7.5, -7.5]]], &device);
        let input_y = Tensor::<TestBackend, 3>::from_floats([[[2.0, 2.0, 3.0, 3.0]]], &device);
        let output = model.forward(input_x, input_y);

        // Expected: fmod operation where sign follows dividend
        let expected = TensorData::from([[[1.3000002f32, -1.3000002, 1.5, -1.5]]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mod_broadcast() {
        // Test broadcasting with fmod=1
        let device = Default::default();
        let model: mod_broadcast_fixed::Model<TestBackend> =
            mod_broadcast_fixed::Model::new(&device);

        let input_x = Tensor::<TestBackend, 2>::from_floats(
            [
                [5.0, -7.0, 8.0, -9.0],
                [4.0, -6.0, 10.0, -11.0],
                [3.0, -5.0, 12.0, -13.0],
            ],
            &device,
        );

        let input_y = Tensor::<TestBackend, 4>::from_floats(
            [
                [[
                    [3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0],
                ]],
                [[
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                ]],
            ],
            &device,
        );

        let output = model.forward(input_x, input_y);

        // Check shape and sample values
        assert_eq!(output.dims(), [2, 1, 3, 4]);

        // Check first batch, first row
        let data = output.to_data();
        let values = data.as_slice::<f32>().unwrap();
        // fmod(5.0, 3.0) = 2.0, fmod(-7.0, 3.0) = -1.0, etc.
        assert!((values[0] - 2.0).abs() < 0.001);
        assert!((values[1] - (-1.0)).abs() < 0.001);
        assert!((values[2] - 2.0).abs() < 0.001);
        assert!((values[3] - 0.0).abs() < 0.001);
    }

    #[test]
    fn mod_shape_shape_and_shape_scalar() {
        // Exercises Mod applied to Shape values (i.e. `[i64; N]` arrays in
        // the generated code), which happens in expanded attention models
        // when head counts are validated via Shape arithmetic.
        let device = Default::default();
        let model: mod_shape::Model<TestBackend> = mod_shape::Model::new(&device);

        let input1 = Tensor::<TestBackend, 3>::zeros([12, 8, 6], &device);
        let input2 = Tensor::<TestBackend, 3>::zeros([4, 2, 3], &device);

        let (shape_mod_shape, shape_mod_scalar) = model.forward(input1, input2);

        // [12, 8, 6] % [4, 2, 3] = [0, 0, 0]
        assert_eq!(shape_mod_shape, [0i64, 0, 0]);
        // [12, 8, 6] % 5 = [2, 3, 1]
        assert_eq!(shape_mod_scalar, [2i64, 3, 1]);
    }

    #[test]
    fn mod_broadcast_remainder() {
        // Test broadcasting with fmod=0 (remainder)
        let device = Default::default();
        let model: mod_broadcast_remainder_fixed::Model<TestBackend> =
            mod_broadcast_remainder_fixed::Model::new(&device);

        let input_x =
            Tensor::<TestBackend, 3>::from_floats([[[7.5], [-8.5], [9.5], [-10.5]]], &device);

        let input_y = Tensor::<TestBackend, 3>::from_floats(
            [
                [[3.0, 4.0, -3.0, -4.0, 5.0]],
                [[3.0, 4.0, -3.0, -4.0, 5.0]],
                [[3.0, 4.0, -3.0, -4.0, 5.0]],
            ],
            &device,
        );

        let output = model.forward(input_x, input_y);

        // Check shape
        assert_eq!(output.dims(), [3, 4, 5]);

        // Check first row values for Python-style remainder
        // remainder(7.5, 3.0) = 1.5, remainder(7.5, 4.0) = 3.5
        // remainder(7.5, -3.0) = -1.5, remainder(7.5, -4.0) = -0.5
        // remainder(7.5, 5.0) = 2.5
        let data = output.to_data();
        let values = data.as_slice::<f32>().unwrap();
        assert!((values[0] - 1.5).abs() < 0.001);
        assert!((values[1] - 3.5).abs() < 0.001);
        assert!((values[2] - (-1.5)).abs() < 0.001);
        assert!((values[3] - (-0.5)).abs() < 0.001);
        assert!((values[4] - 2.5).abs() < 0.001);
    }
}
