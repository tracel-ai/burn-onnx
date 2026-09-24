// Import the shared macro
use crate::include_models;
include_models!(
    greater_or_equal,
    greater_or_equal_broadcast,
    greater_or_equal_scalar,
    greater_or_equal_shape_broadcast,
    greater_or_equal_shape_rank_lift
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn::tensor::{Device, Int, Tensor, TensorData};

    #[test]
    fn greater_or_equal() {
        let device = Default::default();
        let model: greater_or_equal::Model = greater_or_equal::Model::new(&device);

        let input1 = Tensor::<2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, false, true, true]]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn greater_or_equal_scalar() {
        let device = Default::default();
        let model: greater_or_equal_scalar::Model = greater_or_equal_scalar::Model::new(&device);

        let input1 = Tensor::<2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, true, true, false]]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn greater_or_equal_broadcast() {
        let device = Default::default();
        let model: greater_or_equal_broadcast::Model =
            greater_or_equal_broadcast::Model::new(&device);

        // Shape [4, 1] vs [1, 4]
        let input1 = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device);
        let input2 = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([
            [true, false, false, false],
            [true, true, false, false],
            [true, true, true, false],
            [true, true, true, true],
        ]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn greater_or_equal_shape_broadcast() {
        let device = Default::default();
        let model: greater_or_equal_shape_broadcast::Model =
            greater_or_equal_shape_broadcast::Model::default();

        let input_1d = Tensor::<1>::zeros([3], &device);
        let input_4d = Tensor::<4>::zeros([2, 30, 3, 5], &device);

        let (lhs_bc, rhs_bc) = model.forward(input_1d, input_4d);

        assert_eq!(lhs_bc, [1i64, 0, 1, 0]);
        assert_eq!(rhs_bc, [0i64, 1, 1, 1]);
    }

    #[test]
    fn greater_or_equal_shape_operand_with_rank4_tensor() {
        // dim0 of x's shape (B = 2) is a length 1 Shape, lifted to rank 4 before the comparison.
        // Outputs are `x >= 2` and `2 >= x`.
        let device = Default::default();
        let model: greater_or_equal_shape_rank_lift::Model =
            greater_or_equal_shape_rank_lift::Model::from_file(
                concat!(
                    env!("OUT_DIR"),
                    "/model/greater_or_equal_shape_rank_lift.bpk"
                ),
                &device,
            );

        let values: Vec<i64> = (0..120).collect();
        let x = Tensor::<4, Int>::from_data(
            TensorData::new(values.clone(), [2, 3, 4, 5]),
            (&device, burn::tensor::DType::I64),
        );
        let (tensor_shape, shape_tensor) = model.forward(x);

        let expected1: Vec<bool> = values.iter().map(|&v| v >= 2).collect();
        let expected2: Vec<bool> = values.iter().map(|&v| 2 >= v).collect();
        tensor_shape
            .to_data()
            .assert_eq(&TensorData::new(expected1, [2, 3, 4, 5]), false);
        shape_tensor
            .to_data()
            .assert_eq(&TensorData::new(expected2, [2, 3, 4, 5]), false);
    }
}
