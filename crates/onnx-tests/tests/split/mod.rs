use crate::include_models;
include_models!(split, split_uneven, split_axis1);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn split() {
        let device = Default::default();
        let model = split::Model::<TestBackend>::new(&device);
        let shape = [5, 2];
        let input = Tensor::ones(shape, &device);

        let (tensor_1, tensor_2, tensor_3) = model.forward(input);

        assert_eq!(tensor_1.shape(), Shape::from([2, 2]));
        assert_eq!(tensor_2.shape(), Shape::from([2, 2]));
        assert_eq!(tensor_3.shape(), Shape::from([1, 2]));
    }

    #[test]
    fn split_uneven() {
        // num_outputs=3 on dim=10 (not evenly divisible)
        // ONNX spec: ceil(10/3) = 4, splits are [4, 4, 2]
        let device = Default::default();
        let model = split_uneven::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(10, 3).astype(np.float32)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857],
                [1.5230298, -0.23415338, -0.23413695],
                [1.5792128, 0.7674347, -0.46947438],
                [0.54256004, -0.46341768, -0.46572974],
                [0.24196227, -1.9132802, -1.7249179],
                [-0.5622875, -1.0128311, 0.31424734],
                [-0.9080241, -1.4123037, 1.4656488],
                [-0.2257763, 0.0675282, -1.4247482],
                [-0.54438275, 0.11092259, -1.1509936],
                [0.37569803, -0.6006387, -0.29169375],
            ],
            &device,
        );
        let (y0, y1, y2) = model.forward(input);

        assert_eq!(y0.shape(), Shape::from([4, 3]));
        assert_eq!(y1.shape(), Shape::from([4, 3]));
        assert_eq!(y2.shape(), Shape::from([2, 3]));

        let expected_y0 = TensorData::from([
            [0.49671414f32, -0.1382643, 0.64768857],
            [1.5230298, -0.23415338, -0.23413695],
            [1.5792128, 0.7674347, -0.46947438],
            [0.54256004, -0.46341768, -0.46572974],
        ]);
        let expected_y2 = TensorData::from([
            [-0.54438275f32, 0.11092259, -1.1509936],
            [0.37569803, -0.6006387, -0.29169375],
        ]);

        y0.to_data().assert_eq(&expected_y0, true);
        y2.to_data().assert_eq(&expected_y2, true);
    }

    #[test]
    fn split_axis1() {
        // Explicit split sizes [2, 3] on axis=1
        let device = Default::default();
        let model = split_axis1::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(3, 5).astype(np.float32)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857, 1.5230298, -0.23415338],
                [-0.23413695, 1.5792128, 0.7674347, -0.46947438, 0.54256004],
                [-0.46341768, -0.46572974, 0.24196227, -1.9132802, -1.7249179],
            ],
            &device,
        );
        let (y0, y1) = model.forward(input);

        assert_eq!(y0.shape(), Shape::from([3, 2]));
        assert_eq!(y1.shape(), Shape::from([3, 3]));

        let expected_y0 = TensorData::from([
            [0.49671414f32, -0.1382643],
            [-0.23413695, 1.5792128],
            [-0.46341768, -0.46572974],
        ]);
        let expected_y1 = TensorData::from([
            [0.64768857f32, 1.5230298, -0.23415338],
            [0.7674347, -0.46947438, 0.54256004],
            [0.24196227, -1.9132802, -1.7249179],
        ]);

        y0.to_data().assert_eq(&expected_y0, true);
        y1.to_data().assert_eq(&expected_y1, true);
    }
}
