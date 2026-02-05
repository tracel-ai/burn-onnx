// Import the shared macro
use crate::include_models;
include_models!(min, min_broadcast);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn min() {
        let device = Default::default();

        let model: min::Model<TestBackend> = min::Model::new(&device);
        let input1 = Tensor::<TestBackend, 2>::from_floats([[-1.0, 42.0, 0.0, 42.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[2.0, 4.0, 42.0, 25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[-1.0f32, 4.0, 0.0, 25.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn min_broadcast() {
        let device = Default::default();

        let model: min_broadcast::Model<TestBackend> = min_broadcast::Model::new(&device);

        let x_3d = Tensor::<TestBackend, 3>::from_floats(
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
        let y_2d = Tensor::<TestBackend, 2>::from_floats(
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
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 10.0, 10.0],
            ],
            [
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
        ]);

        // Both directions should produce the same result (min is commutative)
        result1.to_data().assert_eq(&expected, true);
        result2.to_data().assert_eq(&expected, true);
    }
}
