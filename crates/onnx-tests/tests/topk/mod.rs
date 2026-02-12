use crate::include_models;
include_models!(
    topk,
    topk_axis0,
    topk_1d,
    topk_3d,
    topk_k_full,
    topk_negative_axis
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn topk() {
        let device = Default::default();
        let model = topk::Model::<TestBackend>::new(&device);

        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.33669037, 0.12880941, 0.23446237, 0.23033303, -1.12285638],
                [-0.18632829, 2.20820141, -0.63799703, 0.46165723, 0.26735088],
                [0.53490466, 0.80935723, 1.11029029, -1.68979895, -0.98895991],
            ],
            &device,
        );
        let (values_tensor, indices_tensor) = model.forward(input);

        let expected_values_tensor = TensorData::from([
            [0.33669037f32, 0.23446237],
            [2.208_201_4, 0.46165723],
            [1.110_290_3, 0.809_357_2],
        ]);
        let expected_indices_tensor = TensorData::from([[0i64, 2], [1, 3], [2, 1]]);

        values_tensor
            .to_data()
            .assert_eq(&expected_values_tensor, true);
        indices_tensor
            .to_data()
            .assert_eq(&expected_indices_tensor, true);
    }

    #[test]
    fn topk_axis0() {
        // axis=0, shape [5, 4], k=3
        let device = Default::default();
        let model = topk_axis0::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(5, 4).astype(np.float32)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857, 1.5230298],
                [-0.23415338, -0.23413695, 1.5792128, 0.7674347],
                [-0.46947438, 0.54256004, -0.46341768, -0.46572974],
                [0.24196227, -1.9132802, -1.7249179, -0.5622875],
                [-1.0128311, 0.31424734, -0.9080241, -1.4123037],
            ],
            &device,
        );
        let (values, indices) = model.forward(input);

        let expected_values = TensorData::from([
            [0.49671414f32, 0.54256004, 1.5792128, 1.5230298],
            [0.24196227, 0.31424734, 0.64768857, 0.7674347],
            [-0.23415338, -0.1382643, -0.46341768, -0.46572974],
        ]);
        let expected_indices = TensorData::from([[0i64, 2, 1, 0], [3, 4, 0, 1], [1, 0, 2, 2]]);

        values.to_data().assert_eq(&expected_values, true);
        indices.to_data().assert_eq(&expected_indices, true);
    }

    #[test]
    fn topk_1d() {
        // 1D tensor [8], axis=0, k=3
        let device = Default::default();
        let model = topk_1d::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(8).astype(np.float32)
        let input = Tensor::<TestBackend, 1>::from_floats(
            [
                0.49671414,
                -0.1382643,
                0.64768857,
                1.5230298,
                -0.23415338,
                -0.23413695,
                1.5792128,
                0.7674347,
            ],
            &device,
        );
        let (values, indices) = model.forward(input);

        let expected_values = TensorData::from([1.5792128f32, 1.5230298, 0.7674347]);
        let expected_indices = TensorData::from([6i64, 3, 7]);

        values.to_data().assert_eq(&expected_values, true);
        indices.to_data().assert_eq(&expected_indices, true);
    }

    #[test]
    fn topk_3d() {
        // axis=1 on [2, 4, 3] tensor, k=2
        let device = Default::default();
        let model = topk_3d::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(2, 4, 3).astype(np.float32)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0.49671414, -0.1382643, 0.64768857],
                    [1.5230298, -0.23415338, -0.23413695],
                    [1.5792128, 0.7674347, -0.46947438],
                    [0.54256004, -0.46341768, -0.46572974],
                ],
                [
                    [0.24196227, -1.9132802, -1.7249179],
                    [-0.5622875, -1.0128311, 0.31424734],
                    [-0.9080241, -1.4123037, 1.4656488],
                    [-0.2257763, 0.0675282, -1.4247482],
                ],
            ],
            &device,
        );
        let (values, indices) = model.forward(input);

        let expected_values = TensorData::from([
            [
                [1.5792128f32, 0.7674347, 0.64768857],
                [1.5230298, -0.1382643, -0.23413695],
            ],
            [
                [0.24196227, 0.0675282, 1.4656488],
                [-0.2257763, -1.0128311, 0.31424734],
            ],
        ]);
        let expected_indices =
            TensorData::from([[[2i64, 2, 0], [1, 0, 1]], [[0, 3, 2], [3, 1, 1]]]);

        values.to_data().assert_eq(&expected_values, true);
        indices.to_data().assert_eq(&expected_indices, true);
    }

    #[test]
    fn topk_k_full() {
        // k = full dimension size (k=5 on axis=1, shape [3, 5])
        let device = Default::default();
        let model = topk_k_full::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(3, 5).astype(np.float32)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857, 1.5230298, -0.23415338],
                [-0.23413695, 1.5792128, 0.7674347, -0.46947438, 0.54256004],
                [-0.46341768, -0.46572974, 0.24196227, -1.9132802, -1.7249179],
            ],
            &device,
        );
        let (values, indices) = model.forward(input);

        let expected_values = TensorData::from([
            [
                1.5230298f32,
                0.64768857,
                0.49671414,
                -0.1382643,
                -0.23415338,
            ],
            [1.5792128, 0.7674347, 0.54256004, -0.23413695, -0.46947438],
            [0.24196227, -0.46341768, -0.46572974, -1.7249179, -1.9132802],
        ]);
        let expected_indices =
            TensorData::from([[3i64, 2, 0, 1, 4], [1, 2, 4, 0, 3], [2, 0, 1, 4, 3]]);

        values.to_data().assert_eq(&expected_values, true);
        indices.to_data().assert_eq(&expected_indices, true);
    }

    #[test]
    fn topk_negative_axis() {
        // axis=-2 on [3, 4, 5] tensor, k=2 (resolves to axis=1)
        let device = Default::default();
        let model = topk_negative_axis::Model::<TestBackend>::new(&device);

        // np.random.seed(42); np.random.randn(3, 4, 5).astype(np.float32)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0.49671414, -0.1382643, 0.64768857, 1.5230298, -0.23415338],
                    [-0.23413695, 1.5792128, 0.7674347, -0.46947438, 0.54256004],
                    [-0.46341768, -0.46572974, 0.24196227, -1.9132802, -1.7249179],
                    [-0.5622875, -1.0128311, 0.31424734, -0.9080241, -1.4123037],
                ],
                [
                    [1.4656488, -0.2257763, 0.0675282, -1.4247482, -0.54438275],
                    [0.11092259, -1.1509936, 0.37569803, -0.6006387, -0.29169375],
                    [-0.6017066, 1.8522782, -0.01349723, -1.0577109, 0.82254493],
                    [-1.2208437, 0.2088636, -1.9596701, -1.328186, 0.19686124],
                ],
                [
                    [0.73846656, 0.17136829, -0.11564828, -0.30110368, -1.478522],
                    [-0.7198442, -0.46063876, 1.0571222, 0.3436183, -1.7630402],
                    [0.32408398, -0.38508227, -0.676922, 0.6116763, 1.0309995],
                    [0.93128014, -0.83921754, -0.3092124, 0.33126342, 0.9755451],
                ],
            ],
            &device,
        );
        let (values, indices) = model.forward(input);

        let expected_values = TensorData::from([
            [
                [0.49671414f32, 1.5792128, 0.7674347, 1.5230298, 0.54256004],
                [
                    -0.23413695,
                    -0.1382643,
                    0.64768857,
                    -0.46947438,
                    -0.23415338,
                ],
            ],
            [
                [1.4656488, 1.8522782, 0.37569803, -0.6006387, 0.82254493],
                [0.11092259, 0.2088636, 0.0675282, -1.0577109, 0.19686124],
            ],
            [
                [0.93128014, 0.17136829, 1.0571222, 0.6116763, 1.0309995],
                [0.73846656, -0.38508227, -0.11564828, 0.3436183, 0.9755451],
            ],
        ]);
        let expected_indices = TensorData::from([
            [[0i64, 1, 1, 0, 1], [1, 0, 0, 1, 0]],
            [[0, 2, 1, 1, 2], [1, 3, 0, 2, 3]],
            [[3, 0, 1, 2, 2], [0, 2, 0, 1, 3]],
        ]);

        values.to_data().assert_eq(&expected_values, true);
        indices.to_data().assert_eq(&expected_indices, true);
    }
}
