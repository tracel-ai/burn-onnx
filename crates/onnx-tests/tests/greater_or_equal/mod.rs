// Import the shared macro
use crate::include_models;
include_models!(
    greater_or_equal,
    greater_or_equal_broadcast,
    greater_or_equal_scalar,
    greater_or_equal_shape_broadcast
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn greater_or_equal() {
        let device = Default::default();
        let model: greater_or_equal::Model = greater_or_equal::Model::new(&device);

        let input1 = Tensor::<2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, false, true, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn greater_or_equal_scalar() {
        let device = Default::default();
        let model: greater_or_equal_scalar::Model = greater_or_equal_scalar::Model::new(&device);

        let input1 = Tensor::<2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, true, true, false]]);

        output.to_data().assert_eq(&expected, true);
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

        output.to_data().assert_eq(&expected, true);
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
}
