// Import the shared macro
use crate::include_models;
include_models!(
    equal,
    equal_scalar,
    equal_shape,
    equal_shape_broadcast,
    equal_two_shapes
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn equal_scalar_to_scalar_and_tensor_to_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: equal::Model = equal::Model::default();

        // Run the model
        let input = Tensor::<4>::from_floats([[[[1., 1., 1., 1.]]]], &Default::default());

        let scalar = 2f64;
        let (tensor_out, scalar_out) = model.forward(input, scalar);
        let expected_tensor = TensorData::from([[[[true, true, true, true]]]]);
        let expected_scalar = false;

        tensor_out.to_data().assert_eq(&expected_tensor, true);
        assert_eq!(scalar_out, expected_scalar);
    }

    #[test]
    fn equal_shape() {
        // Test comparing a Shape output with a constant shape
        let model: equal_shape::Model = equal_shape::Model::default();

        // Create input tensor with shape [2, 3, 4]
        let input = Tensor::<3>::zeros([2, 3, 4], &Default::default());

        let output = model.forward(input);
        // Shape [2, 3, 4] should equal [2, 3, 4]
        let expected = TensorData::from([true, true, true]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn equal_scalar() {
        let device = Default::default();
        let model: equal_scalar::Model = equal_scalar::Model::new(&device);

        let x = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 2.0]], &device);
        let y = 2.0f32;

        let (tensor_scalar, scalar_tensor) = model.forward(x, y);
        let expected = TensorData::from([[false, true, false, true]]);

        tensor_scalar.to_data().assert_eq(&expected, true);
        scalar_tensor.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn equal_two_shapes() {
        // Test comparing shapes from two different tensors
        let model: equal_two_shapes::Model = equal_two_shapes::Model::default();

        // Create two input tensors with same shape [2, 3, 4]
        let input1 = Tensor::<3>::zeros([2, 3, 4], &Default::default());
        let input2 = Tensor::<3>::ones([2, 3, 4], &Default::default());

        let output = model.forward(input1, input2);
        // Both have shape [2, 3, 4] so all elements should be equal (1 for true)
        let expected: [i64; 3] = [1, 1, 1];

        assert_eq!(output, expected);
    }

    #[test]
    fn equal_shape_broadcast() {
        let device = Default::default();
        let model: equal_shape_broadcast::Model = equal_shape_broadcast::Model::default();

        let input_1d = Tensor::<1>::zeros([3], &device);
        let input_4d = Tensor::<4>::zeros([2, 30, 3, 5], &device);

        let (lhs_bc, rhs_bc) = model.forward(input_1d, input_4d);

        assert_eq!(lhs_bc, [0i64, 0, 1, 0]);
        assert_eq!(rhs_bc, [0i64, 0, 1, 0]);
    }
}
