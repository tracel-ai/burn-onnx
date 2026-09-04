// Import the shared macro
use crate::include_models;
include_models!(max, max_broadcast, max_scalar, max_shape, max_shape_tensor);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    #[test]
    fn max() {
        let device = Default::default();

        let model: max::Model = max::Model::new(&device);
        let input1 = Tensor::<2>::from_floats([[1.0, 42.0, 9.0, 42.0]], &device);
        let input2 = Tensor::<2>::from_floats([[42.0, 4.0, 42.0, 25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[42.0f32, 42.0, 42.0, 42.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn max_broadcast() {
        let device = Default::default();

        let model: max_broadcast::Model = max_broadcast::Model::new(&device);

        let x_3d = Tensor::<3>::from_floats(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                ],
            ],
            &device,
        );
        let y_2d = Tensor::<2>::from_floats(
            [
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            &device,
        );
        let a_2d = y_2d.clone();
        let b_3d = x_3d.clone();

        let (result1, result2) = model.forward(x_3d, y_2d, a_2d, b_3d);

        let expected = TensorData::from([
            [
                [10.0f32, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]);

        // Both directions should produce the same result (max is commutative)
        result1.to_data().assert_eq(&expected, true);
        result2.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn max_scalar() {
        let device = Default::default();
        let model: max_scalar::Model = max_scalar::Model::default();

        let tensor = Tensor::<2>::from_floats([[0., 1.5, -3., 2.5], [4., 1., -1., 3.5]], &device);
        let (scalar_tensor, tensor_scalar, scalar_scalar) = model.forward(1., tensor, 2.);

        scalar_tensor.to_data().assert_eq(
            &TensorData::from([[1.0f32, 1.5, 1., 2.5], [4., 1., 1., 3.5]]),
            true,
        );
        tensor_scalar.to_data().assert_eq(
            &TensorData::from([[2.0f32, 2., 2., 2.5], [4., 2., 2., 3.5]]),
            true,
        );
        assert_eq!(scalar_scalar, 2.);
    }

    #[test]
    fn max_shape() {
        let device = Default::default();
        let model: max_shape::Model = max_shape::Model::default();

        let input1 = Tensor::<3>::ones([10, 8, 6], &device);
        let input2 = Tensor::<3>::ones([2, 30, 4], &device);
        let (shape_scalar, scalar_shape, shape_shape) = model.forward(input1, input2);

        // there is a constant node with 7
        assert_eq!(shape_scalar, [10, 8, 7]);
        assert_eq!(scalar_shape, [10, 8, 7]);
        assert_eq!(shape_shape, [10, 30, 6]);
    }

    #[test]
    fn max_shape_tensor() {
        let device = Default::default();
        let model: max_shape_tensor::Model = max_shape_tensor::Model::default();

        let input_tensor = Tensor::<3>::ones([5, 7, 9], &device);
        let input_3d = Tensor::<3, Int>::from_data(
            TensorData::from([
                [[1i64, 9, 2], [6, 3, 12], [5, 7, 9], [0, 0, 0]],
                [[10, 2, 3], [4, 20, 8], [7, 7, 11], [2, 8, 15]],
            ]),
            (&device, DType::I64),
        );
        let (shape_tensor, tensor_shape) = model.forward(input_tensor, input_3d);

        let expected = TensorData::from([
            [[5i64, 9, 9], [6, 7, 12], [5, 7, 9], [5, 7, 9]],
            [[10, 7, 9], [5, 20, 9], [7, 7, 11], [5, 8, 15]],
        ]);
        shape_tensor.to_data().assert_eq(&expected, true);
        tensor_shape.to_data().assert_eq(&expected, true);
    }
}
