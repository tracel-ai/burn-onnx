use crate::include_models;
include_models!(
    gather_elements,
    gather_elements_axis0,
    gather_elements_3d,
    gather_elements_negative_axis
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn gather_elements() {
        let model: gather_elements::Model<TestBackend> = gather_elements::Model::default();

        let device = Default::default();
        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2.], [3., 4.]], &device);
        let index = Tensor::<TestBackend, 2, Int>::from_ints([[0, 0], [1, 0]], &device);
        let output = model.forward(input, index);
        let expected = TensorData::from([[1f32, 1.], [4., 3.]]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_elements_axis0() {
        // axis=0, data [3, 3], indices [2, 3]
        let device = Default::default();
        let model = gather_elements_axis0::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(3, 3).astype(np.float32)
        let data = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857],
                [1.5230298, -0.23415338, -0.23413695],
                [1.5792128, 0.7674347, -0.46947438],
            ],
            &device,
        );
        let indices = Tensor::<TestBackend, 2, Int>::from_ints([[0, 2, 1], [2, 0, 0]], &device);
        let output = model.forward(data, indices);

        let expected = TensorData::from([
            [0.49671414f32, 0.7674347, -0.23413695],
            [1.5792128, -0.1382643, 0.64768857],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn gather_elements_3d() {
        // axis=1, data [2, 3, 4], indices [2, 2, 4]
        let device = Default::default();
        let model = gather_elements_3d::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(2, 3, 4).astype(np.float32)
        let data = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0.49671414, -0.1382643, 0.64768857, 1.5230298],
                    [-0.23415338, -0.23413695, 1.5792128, 0.7674347],
                    [-0.46947438, 0.54256004, -0.46341768, -0.46572974],
                ],
                [
                    [0.24196227, -1.9132802, -1.7249179, -0.5622875],
                    [-1.0128311, 0.31424734, -0.9080241, -1.4123037],
                    [1.4656488, -0.2257763, 0.0675282, -1.4247482],
                ],
            ],
            &device,
        );
        let indices = Tensor::<TestBackend, 3, Int>::from_ints(
            [[[0, 2, 0, 2], [2, 0, 0, 2]], [[1, 0, 1, 1], [1, 0, 1, 0]]],
            &device,
        );
        let output = model.forward(data, indices);

        let expected = TensorData::from([
            [
                [0.49671414f32, 0.54256004, 0.64768857, -0.46572974],
                [-0.46947438, -0.1382643, 0.64768857, -0.46572974],
            ],
            [
                [-1.0128311, -1.9132802, -0.9080241, -1.4123037],
                [-1.0128311, -1.9132802, -0.9080241, -0.5622875],
            ],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn gather_elements_negative_axis() {
        // axis=-1 (resolves to 1), data [3, 4], indices [3, 4]
        let device = Default::default();
        let model = gather_elements_negative_axis::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(3, 4).astype(np.float32)
        let data = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857, 1.5230298],
                [-0.23415338, -0.23413695, 1.5792128, 0.7674347],
                [-0.46947438, 0.54256004, -0.46341768, -0.46572974],
            ],
            &device,
        );
        let indices = Tensor::<TestBackend, 2, Int>::from_ints(
            [[3, 1, 1, 0], [3, 0, 0, 2], [2, 2, 1, 3]],
            &device,
        );
        let output = model.forward(data, indices);

        let expected = TensorData::from([
            [1.5230298f32, -0.1382643, -0.1382643, 0.49671414],
            [0.7674347, -0.23415338, -0.23415338, 1.5792128],
            [-0.46341768, -0.46341768, 0.54256004, -0.46572974],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}
