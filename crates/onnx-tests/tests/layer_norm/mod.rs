// Import the shared macro
use crate::include_models;
include_models!(
    layer_norm,
    layer_norm_no_bias,
    layer_norm_custom_epsilon,
    layer_norm_4d,
    layer_norm_runtime_mean,
    layer_norm_broadcast
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn layer_norm() {
        let device = Default::default();
        let model: layer_norm::Model = layer_norm::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<3>::from_floats(
            [
                [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                [
                    [12., 13., 14., 15.],
                    [16., 17., 18., 19.],
                    [20., 21., 22., 23.],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [
                [-1.3416f32, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ],
            [
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_no_bias() {
        // LayerNorm without bias (2 inputs: X + scale only)
        let device = Default::default();
        let model: layer_norm_no_bias::Model = layer_norm_no_bias::Model::default();

        let input = Tensor::<3>::from_floats(
            [
                [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                [
                    [12., 13., 14., 15.],
                    [16., 17., 18., 19.],
                    [20., 21., 22., 23.],
                ],
            ],
            &device,
        );
        let output = model.forward(input);

        // Same as with-bias case since bias=0 and scale=1
        let expected = TensorData::from([
            [
                [-1.3416f32, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ],
            [
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_custom_epsilon() {
        // LayerNorm with epsilon=0.001 (larger than default 1e-5)
        let device = Default::default();
        let model: layer_norm_custom_epsilon::Model = layer_norm_custom_epsilon::Model::default();

        let input = Tensor::<3>::from_floats(
            [
                [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                [
                    [12., 13., 14., 15.],
                    [16., 17., 18., 19.],
                    [20., 21., 22., 23.],
                ],
            ],
            &device,
        );
        let output = model.forward(input);

        // Slightly different values due to larger epsilon
        let expected = TensorData::from([
            [
                [-1.3411f32, -0.4470, 0.4470, 1.3411],
                [-1.3411, -0.4470, 0.4470, 1.3411],
                [-1.3411, -0.4470, 0.4470, 1.3411],
            ],
            [
                [-1.3411, -0.4470, 0.4470, 1.3411],
                [-1.3411, -0.4470, 0.4470, 1.3411],
                [-1.3411, -0.4470, 0.4470, 1.3411],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_4d() {
        // 4D input [2, 2, 3, 4], axis=-1
        let device = Default::default();
        let model: layer_norm_4d::Model = layer_norm_4d::Model::default();

        let input = Tensor::<4>::from_floats(
            [
                [
                    [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                    [
                        [12., 13., 14., 15.],
                        [16., 17., 18., 19.],
                        [20., 21., 22., 23.],
                    ],
                ],
                [
                    [
                        [24., 25., 26., 27.],
                        [28., 29., 30., 31.],
                        [32., 33., 34., 35.],
                    ],
                    [
                        [36., 37., 38., 39.],
                        [40., 41., 42., 43.],
                        [44., 45., 46., 47.],
                    ],
                ],
            ],
            &device,
        );
        let output = model.forward(input);

        let expected = TensorData::from([
            [
                [
                    [-1.3416f32, -0.4472, 0.4472, 1.3416],
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                ],
                [
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                ],
            ],
            [
                [
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                ],
                [
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                    [-1.3416, -0.4472, 0.4472, 1.3416],
                ],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_runtime_scale_with_mean() {
        let device = Default::default();
        let model: layer_norm_runtime_mean::Model = layer_norm_runtime_mean::Model::new(&device);
        let x = burn::tensor::Tensor::<1, burn::tensor::Int>::arange(0..12, &device)
            .float()
            .powf_scalar(1.5)
            .reshape([2, 2, 3]);
        let scale =
            burn::tensor::Tensor::<2>::from_floats([[1.0, 2.0, 0.5], [-1.0, 1.5, 3.0]], &device);

        let (y, mean) = model.forward(x, scale);

        let tolerance = burn::tensor::Tolerance::absolute(1e-4);
        y.to_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([
                [
                    [-1.197_790_1f32, -1.885_971_3, -0.238_547_06],
                    [-0.126_213_03, 1.260_969_3, 4.953_033_4],
                ],
                [
                    [-1.403_542_9, -1.781_185_5, -0.169_781_19],
                    [-0.247_078_57, 1.300_929_4, 4.558_000_6],
                ],
            ]),
            tolerance,
        );
        mean.to_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([[[4.700_819_5f32]], [[25.158_377]]]),
            tolerance,
        );
    }

    #[test]
    fn layer_norm_broadcast_scale_and_bias() {
        let device = Default::default();
        let model = layer_norm_broadcast::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/layer_norm_broadcast.bpk"),
            &device,
        );
        let x = burn::tensor::Tensor::<1, burn::tensor::Int>::arange(0..12, &device)
            .float()
            .powf_scalar(1.5)
            .reshape([2, 2, 3]);

        let y = model.forward(x);

        y.to_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([
                [
                    [-0.697_79f32, -1.885_971, -0.022_906],
                    [0.626_213, 1.681_292, -2.151_011],
                ],
                [
                    [-0.903_543, -1.781_186, -0.160_438],
                    [0.747_079, 1.734_572, -2.019_334],
                ],
            ]),
            burn::tensor::Tolerance::absolute(1e-4),
        );
    }
}
