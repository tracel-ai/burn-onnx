// Import the shared macro
use crate::include_models;
include_models!(
    layer_norm,
    layer_norm_no_bias,
    layer_norm_custom_epsilon,
    layer_norm_4d
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn layer_norm() {
        let device = Default::default();
        let model: layer_norm::Model<TestBackend> = layer_norm::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 3>::from_floats(
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
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_no_bias() {
        // LayerNorm without bias (2 inputs: X + scale only)
        let device = Default::default();
        let model: layer_norm_no_bias::Model<TestBackend> = layer_norm_no_bias::Model::default();

        let input = Tensor::<TestBackend, 3>::from_floats(
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
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_custom_epsilon() {
        // LayerNorm with epsilon=0.001 (larger than default 1e-5)
        let device = Default::default();
        let model: layer_norm_custom_epsilon::Model<TestBackend> =
            layer_norm_custom_epsilon::Model::default();

        let input = Tensor::<TestBackend, 3>::from_floats(
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
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_4d() {
        // 4D input [2, 2, 3, 4], axis=-1
        let device = Default::default();
        let model: layer_norm_4d::Model<TestBackend> = layer_norm_4d::Model::default();

        let input = Tensor::<TestBackend, 4>::from_floats(
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
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
