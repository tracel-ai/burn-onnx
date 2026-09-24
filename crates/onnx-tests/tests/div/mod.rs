// Import the shared macro
use crate::include_models;
include_models!(
    div,
    div_broadcast,
    div_shape,
    div_shape_broadcast,
    div_shape_rank_lift,
    div_shape_tensor
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn::tensor::{DType, Device, Int, Tensor, TensorData, Tolerance};

    #[test]
    fn div_tensor_by_scalar_and_tensor_by_tensor() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: div::Model = div::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats([[[[3., 6., 6., 9.]]]], &device);
        let scalar1 = 9.0f64;
        let scalar2 = 3.0f64;
        let output = model.forward(input, scalar1, scalar2);
        let expected = TensorData::from([[[[1f32, 2., 2., 3.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn div_shape_with_scalar_and_shape() {
        // Initialize the model
        let device = Default::default();
        let model: div_shape::Model = div_shape::Model::new(&device);

        // Create input tensors
        let input1 = Tensor::<3>::ones([8, 12, 16], &device);
        let input2 = Tensor::<3>::ones([2, 3, 4], &device);
        let (shape_div_scalar, shape_div_shape) = model.forward(input1, input2);

        // Expected outputs
        let expected_scalar = [4, 6, 8]; // shape1 [8, 12, 16] / 2
        let expected_shape = [4, 4, 4]; // shape1 [8, 12, 16] / shape2 [2, 3, 4]

        assert_eq!(shape_div_scalar, expected_scalar);
        assert_eq!(shape_div_shape, expected_shape);
    }

    #[test]
    fn div_broadcast_tensor_ranks() {
        let model: div_broadcast::Model = div_broadcast::Model::default();
        let device = Default::default();

        let x_3d = Tensor::<3>::from_floats(
            [
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                    [90.0, 100.0, 110.0, 120.0],
                ],
                [
                    [12.0, 24.0, 36.0, 48.0],
                    [60.0, 72.0, 84.0, 96.0],
                    [108.0, 120.0, 132.0, 144.0],
                ],
            ],
            &device,
        );

        let y_2d = Tensor::<2>::from_floats(
            [
                [2.0, 4.0, 6.0, 8.0],
                [10.0, 12.0, 14.0, 16.0],
                [18.0, 20.0, 22.0, 24.0],
            ],
            &device,
        );

        let a_2d = Tensor::<2>::from_floats(
            [
                [100.0, 200.0, 300.0, 400.0],
                [500.0, 600.0, 700.0, 800.0],
                [900.0, 1000.0, 1100.0, 1200.0],
            ],
            &device,
        );

        let b_3d = Tensor::<3>::from_floats(
            [
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                    [90.0, 100.0, 110.0, 120.0],
                ],
                [
                    [5.0, 10.0, 15.0, 20.0],
                    [25.0, 30.0, 35.0, 40.0],
                    [45.0, 50.0, 55.0, 60.0],
                ],
            ],
            &device,
        );

        let (result1, result2) = model.forward(x_3d, y_2d, a_2d, b_3d);

        // Expected outputs from Python evaluation
        let expected1 = TensorData::from([
            [
                [5.0f32, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
            [
                [6.0, 6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0, 6.0],
            ],
        ]);

        let expected2 = TensorData::from([
            [
                [10.0f32, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            [
                [20.0, 20.0, 20.0, 20.0],
                [20.0, 20.0, 20.0, 20.0],
                [20.0, 20.0, 20.0, 20.0],
            ],
        ]);

        assert_eq!(result1.dtype(), DType::F32);
        result1
            .to_data()
            .assert_approx_eq::<f32>(&expected1, Tolerance::default());
        assert_eq!(result2.dtype(), DType::F32);
        result2
            .to_data()
            .assert_approx_eq::<f32>(&expected2, Tolerance::default());
    }

    #[test]
    fn div_shape_broadcast() {
        let device = Default::default();
        let model: div_shape_broadcast::Model = div_shape_broadcast::Model::default();

        let input_1d = Tensor::<1>::zeros([10], &device);
        let input_4d = Tensor::<4>::zeros([3, 7, 24, 45], &device);

        let (lhs_bc, rhs_bc) = model.forward(input_1d, input_4d);

        assert_eq!(lhs_bc, [3i64, 1, 0, 0]);
        assert_eq!(rhs_bc, [0i64, 0, 2, 4]);
    }

    #[test]
    fn div_shape_operand_with_rank4_tensor() {
        // dim0 of x's shape (B = 2) is a length 1 Shape, lifted to rank 4 before the op.
        // Outputs are `x / 2` and `2 / x`.
        let device = Default::default();
        let model: div_shape_rank_lift::Model = div_shape_rank_lift::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/div_shape_rank_lift.bpk"),
            &device,
        );

        let values: Vec<i64> = (1..=120).collect();
        let x = Tensor::<4, Int>::from_data(
            TensorData::new(values.clone(), [2, 3, 4, 5]),
            (&device, burn::tensor::DType::I64),
        );
        let (tensor_shape, shape_tensor) = model.forward(x);

        let expected1: Vec<i64> = values.iter().map(|&v| v / 2).collect();
        let expected2: Vec<i64> = values.iter().map(|&v| 2 / v).collect();
        tensor_shape
            .to_data()
            .assert_eq(&TensorData::new(expected1, [2, 3, 4, 5]), true);
        shape_tensor
            .to_data()
            .assert_eq(&TensorData::new(expected2, [2, 3, 4, 5]), true);
    }
}
