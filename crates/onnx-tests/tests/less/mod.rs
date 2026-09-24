// Import the shared macro
use crate::include_models;
include_models!(
    less,
    less_scalar,
    less_broadcast,
    less_shape_broadcast,
    less_shape_rank_lift
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn::tensor::{Device, Int, Tensor, TensorData};

    #[test]
    fn less() {
        let device = Default::default();
        let model: less::Model = less::Model::new(&device);

        let input1 = Tensor::<2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, true, false, false]]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn less_scalar() {
        let device = Default::default();
        let model: less_scalar::Model = less_scalar::Model::new(&device);

        let input1 = Tensor::<2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, false, false, true]]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn less_broadcast() {
        let device = Default::default();
        let model: less_broadcast::Model = less_broadcast::Model::new(&device);

        // Shape [1, 77] vs [77, 1] - testing the CLIP-like pattern
        let input1 = Tensor::<2>::from_floats(
            [[0.0, 1.0, -1.0, 2.0, -2.0]], // Using just 5 values for simplicity
            &device,
        );
        let input2 = Tensor::<2>::from_floats(
            [[0.5], [1.5], [-0.5], [-1.5], [2.5]], // 5x1 shape
            &device,
        );

        let output = model.forward(input1, input2);
        // Expected output shape: [5, 5]
        let expected = TensorData::from([
            [true, false, true, false, true],
            [true, true, true, false, true],
            [false, false, true, false, true],
            [false, false, false, false, true],
            [true, true, true, true, true],
        ]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn less_shape_broadcast() {
        let device = Default::default();
        let model: less_shape_broadcast::Model = less_shape_broadcast::Model::default();

        let input_1d = Tensor::<1>::zeros([3], &device);
        let input_4d = Tensor::<4>::zeros([2, 30, 3, 5], &device);

        let (lhs_bc, rhs_bc) = model.forward(input_1d, input_4d);

        assert_eq!(lhs_bc, [0i64, 1, 0, 1]);
        assert_eq!(rhs_bc, [1i64, 0, 0, 0]);
    }

    #[test]
    fn less_shape_operand_with_rank4_tensor() {
        // dim0 of x's shape (B = 2) is a length 1 Shape, lifted to rank 4 before the comparison.
        // Outputs are `x < 2` and `2 < x`.
        let device = Default::default();
        let model: less_shape_rank_lift::Model = less_shape_rank_lift::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/less_shape_rank_lift.bpk"),
            &device,
        );

        let values: Vec<i64> = (0..120).collect();
        let x = Tensor::<4, Int>::from_data(
            TensorData::new(values.clone(), [2, 3, 4, 5]),
            (&device, burn::tensor::DType::I64),
        );
        let (tensor_shape, shape_tensor) = model.forward(x);

        let expected1: Vec<bool> = values.iter().map(|&v| v < 2).collect();
        let expected2: Vec<bool> = values.iter().map(|&v| 2 < v).collect();
        tensor_shape
            .to_data()
            .assert_eq(&TensorData::new(expected1, [2, 3, 4, 5]), false);
        shape_tensor
            .to_data()
            .assert_eq(&TensorData::new(expected2, [2, 3, 4, 5]), false);
    }
}
