// Import the shared macro
use crate::include_models;
include_models!(
    and,
    and_scalar,
    and_scalar_tensor,
    and_broadcast,
    and_shape_broadcast
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    #[test]
    fn and() {
        let device = Default::default();
        let model: and::Model = and::Model::new(&device);

        let input_x = Tensor::<4, Bool>::from_bool(
            TensorData::from([[[[false, false, true, true]]]]),
            &device,
        );
        let input_y = Tensor::<4, Bool>::from_bool(
            TensorData::from([[[[false, true, false, true]]]]),
            &device,
        );

        let output = model.forward(input_x, input_y).to_data();
        let expected = TensorData::from([[[[false, false, false, true]]]]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn and_scalar() {
        let device = Default::default();
        let model: and_scalar::Model = and_scalar::Model::new(&device);

        // Test various combinations of scalar boolean inputs
        // (input1 && true) && (input2 && false) = input1 && false = false
        assert_eq!(model.forward(false, false), false);
        assert_eq!(model.forward(false, true), false);
        assert_eq!(model.forward(true, false), false);
        assert_eq!(model.forward(true, true), false); // true && false = false
    }

    #[test]
    fn and_scalar_tensor() {
        let device = Default::default();
        let model: and_scalar_tensor::Model = and_scalar_tensor::Model::new(&device);

        let input = Tensor::<2, Bool>::from_bool(
            TensorData::from([[true, false, true], [false, true, false]]),
            &device,
        );

        // And(true, input) should equal input
        let output = model.forward(input.clone()).to_data();
        let expected = input.to_data();

        output.assert_eq(&expected, true);
    }

    #[test]
    fn and_broadcast_tensor_ranks() {
        let model = and_broadcast::Model::default();
        let device = Default::default();

        // Create tensors matching the Python script
        let x_3d = Tensor::<3, Bool>::from_data(
            [
                [
                    [true, false, false, false],
                    [true, false, true, false],
                    [false, true, true, true],
                ],
                [
                    [true, false, false, true],
                    [false, false, true, true],
                    [true, true, false, true],
                ],
            ],
            &device,
        );

        let y_2d = Tensor::<2, Bool>::from_data(
            [
                [false, false, true, true],
                [false, true, true, false],
                [false, false, false, true],
            ],
            &device,
        );

        let a_2d = Tensor::<2, Bool>::from_data(
            [
                [false, true, false, false],
                [false, false, false, true],
                [true, false, false, true],
            ],
            &device,
        );

        let b_3d = Tensor::<3, Bool>::from_data(
            [
                [
                    [true, false, true, true],
                    [true, true, false, true],
                    [false, false, false, false],
                ],
                [
                    [true, true, true, false],
                    [false, true, false, false],
                    [false, false, true, false],
                ],
            ],
            &device,
        );

        let (result1, result2) = model.forward(x_3d, y_2d, a_2d, b_3d);

        // Expected outputs from the Python script
        let expected1 = TensorData::from([
            [
                [false, false, false, false],
                [false, false, true, false],
                [false, false, false, true],
            ],
            [
                [false, false, false, true],
                [false, false, true, false],
                [false, false, false, true],
            ],
        ]);
        let expected2 = TensorData::from([
            [
                [false, false, false, false],
                [false, false, false, true],
                [false, false, false, false],
            ],
            [
                [false, true, false, false],
                [false, false, false, false],
                [false, false, false, false],
            ],
        ]);

        result1.to_data().assert_eq(&expected1, true);
        result2.to_data().assert_eq(&expected2, true);
    }

    #[test]
    fn and_shape_broadcast() {
        let device = Default::default();
        let model: and_shape_broadcast::Model = and_shape_broadcast::Model::default();

        let input_a = Tensor::<1>::zeros([3], &device);
        let input_b = Tensor::<1>::zeros([7], &device);
        let input_c = Tensor::<4>::zeros([2, 3, 4, 5], &device);
        let input_d = Tensor::<4>::zeros([9, 1, 4, 2], &device);

        let (lhs_bc, rhs_bc, same) = model.forward(input_a, input_b, input_c, input_d);

        // greater_1 = [0], greater_4 = [0, 1, 0, 1]
        assert_eq!(lhs_bc, [0i64, 0, 0, 0]);
        assert_eq!(rhs_bc, [0i64, 0, 0, 0]);
        assert_eq!(same, [0i64, 1, 0, 1]);
    }
}
