// Import the shared macro
use crate::include_models;
include_models!(max, max_broadcast);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn max() {
        let device = Default::default();

        let model: max::Model<TestBackend> = max::Model::new(&device);
        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 42.0, 9.0, 42.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[42.0, 4.0, 42.0, 25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[42.0f32, 42.0, 42.0, 42.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn max_broadcast() {
        let device = Default::default();

        let model: max_broadcast::Model<TestBackend> = max_broadcast::Model::new(&device);

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
}
