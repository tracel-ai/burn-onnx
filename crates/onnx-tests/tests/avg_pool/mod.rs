// Import the shared macro
use crate::include_models;
include_models!(
    avg_pool1d_dilations,
    avg_pool2d_dilations,
    avg_pool3d_dilations,
    avg_pool1d,
    avg_pool1d_asymmetric_padding,
    avg_pool1d_ceil_mode,
    avg_pool1d_ceil_mode_opset10,
    avg_pool2d,
    avg_pool2d_asymmetric_padding,
    avg_pool2d_same_upper_dynamic,
    avg_pool2d_ceil_mode
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Shape, Tensor, TensorData, Tolerance};

    type PoolExpected<const D: usize> = [(&'static [f32], [usize; D]); 6];

    fn check_dilated_pool<const D: usize>(
        forward: impl Fn(Tensor<D>) -> [Tensor<D>; 6],
        expected: &[(usize, PoolExpected<D>)],
    ) {
        let device = Default::default();
        for (size, outputs) in expected {
            let mut shape = [*size; D];
            shape[..2].fill(1);
            let values = (0..shape.iter().product::<usize>())
                .map(|i| i as f32 / 10.)
                .collect::<alloc::vec::Vec<_>>();
            let input = Tensor::from_data(
                TensorData::new(values, shape),
                (&device, burn::tensor::DType::F32),
            );
            for (actual, (values, shape)) in forward(input).into_iter().zip(outputs) {
                actual.to_data().assert_approx_eq::<f32>(
                    &TensorData::new(values.to_vec(), *shape),
                    Tolerance::absolute(1e-4),
                );
            }
        }
    }

    #[test]
    fn avg_pool1d_dilated() {
        let model = avg_pool1d_dilations::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/avg_pool1d_dilations.bpk"),
            &Default::default(),
        );
        check_dilated_pool(
            |input| {
                let (a, b, c, d, e, f) = model.forward(input);
                [a, b, c, d, e, f]
            },
            &[
                (
                    6,
                    [
                        // exclude_pad_floor
                        (&[0.1, 0.2, 0.4], [1, 1, 3]),
                        // exclude_pad_ceil
                        (&[0.1, 0.2, 0.4, 0.5], [1, 1, 4]),
                        // include_pad_floor
                        (&[0.05, 0.2, 0.4], [1, 1, 3]),
                        // include_pad_ceil
                        (&[0.05, 0.2, 0.4, 0.5], [1, 1, 4]),
                        // same_upper
                        (&[0.1, 0.3, 0.4], [1, 1, 3]),
                        // same_lower
                        (&[0.1, 0.2, 0.4], [1, 1, 3]),
                    ],
                ),
                (
                    7,
                    [
                        // exclude_pad_floor
                        (&[0.1, 0.2, 0.4, 0.5], [1, 1, 4]),
                        // exclude_pad_ceil
                        (&[0.1, 0.2, 0.4, 0.5], [1, 1, 4]),
                        // include_pad_floor
                        (&[0.05, 0.2, 0.4, 0.25], [1, 1, 4]),
                        // include_pad_ceil
                        (&[0.05, 0.2, 0.4, 0.25], [1, 1, 4]),
                        // same_upper
                        (&[0.1, 0.2, 0.4, 0.5], [1, 1, 4]),
                        // same_lower
                        (&[0.1, 0.2, 0.4, 0.5], [1, 1, 4]),
                    ],
                ),
            ],
        );
    }

    #[test]
    fn avg_pool2d_dilated() {
        let model = avg_pool2d_dilations::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/avg_pool2d_dilations.bpk"),
            &Default::default(),
        );
        check_dilated_pool(
            |input| {
                let (a, b, c, d, e, f) = model.forward(input);
                [a, b, c, d, e, f]
            },
            &[
                (
                    6,
                    [
                        // exclude_pad_floor
                        (&[0.7, 0.8, 1.0, 1.3, 1.4, 1.6, 2.5, 2.6, 2.8], [1, 1, 3, 3]),
                        // exclude_pad_ceil
                        (
                            &[
                                0.7, 0.8, 1.0, 1.1, 1.3, 1.4, 1.6, 1.7, 2.5, 2.6, 2.8, 2.9, 3.1,
                                3.2, 3.4, 3.5,
                            ],
                            [1, 1, 4, 4],
                        ),
                        // include_pad_floor
                        (
                            &[0.175, 0.4, 0.5, 0.65, 1.4, 1.6, 1.25, 2.6, 2.8],
                            [1, 1, 3, 3],
                        ),
                        // include_pad_ceil
                        (
                            &[
                                0.175, 0.4, 0.5, 0.55, 0.65, 1.4, 1.6, 1.7, 1.25, 2.6, 2.8, 2.9,
                                1.55, 3.2, 3.4, 3.5,
                            ],
                            [1, 1, 4, 4],
                        ),
                        // same_upper
                        (&[0.7, 0.9, 1.0, 1.9, 2.1, 2.2, 2.5, 2.7, 2.8], [1, 1, 3, 3]),
                        // same_lower
                        (&[0.7, 0.8, 1.0, 1.3, 1.4, 1.6, 2.5, 2.6, 2.8], [1, 1, 3, 3]),
                    ],
                ),
                (
                    7,
                    [
                        // exclude_pad_floor
                        (
                            &[
                                0.8, 0.9, 1.1, 1.2, 1.5, 1.6, 1.8, 1.9, 2.9, 3.0, 3.2, 3.3, 3.6,
                                3.7, 3.9, 4.0,
                            ],
                            [1, 1, 4, 4],
                        ),
                        // exclude_pad_ceil
                        (
                            &[
                                0.8, 0.9, 1.1, 1.2, 1.5, 1.6, 1.8, 1.9, 2.9, 3.0, 3.2, 3.3, 3.6,
                                3.7, 3.9, 4.0,
                            ],
                            [1, 1, 4, 4],
                        ),
                        // include_pad_floor
                        (
                            &[
                                0.2, 0.45, 0.55, 0.3, 0.75, 1.6, 1.8, 0.95, 1.45, 3.0, 3.2, 1.65,
                                0.9, 1.85, 1.95, 1.0,
                            ],
                            [1, 1, 4, 4],
                        ),
                        // include_pad_ceil
                        (
                            &[
                                0.2, 0.45, 0.55, 0.3, 0.75, 1.6, 1.8, 0.95, 1.45, 3.0, 3.2, 1.65,
                                0.9, 1.85, 1.95, 1.0,
                            ],
                            [1, 1, 4, 4],
                        ),
                        // same_upper
                        (
                            &[
                                0.8, 0.9, 1.1, 1.2, 1.5, 1.6, 1.8, 1.9, 2.9, 3.0, 3.2, 3.3, 3.6,
                                3.7, 3.9, 4.0,
                            ],
                            [1, 1, 4, 4],
                        ),
                        // same_lower
                        (
                            &[
                                0.8, 0.9, 1.1, 1.2, 1.5, 1.6, 1.8, 1.9, 2.9, 3.0, 3.2, 3.3, 3.6,
                                3.7, 3.9, 4.0,
                            ],
                            [1, 1, 4, 4],
                        ),
                    ],
                ),
            ],
        );
    }

    #[test]
    fn avg_pool3d_dilated() {
        let model = avg_pool3d_dilations::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/avg_pool3d_dilations.bpk"),
            &Default::default(),
        );
        check_dilated_pool(
            |input| {
                let (a, b, c, d, e, f) = model.forward(input);
                [a, b, c, d, e, f]
            },
            &[
                (
                    6,
                    [
                        // exclude_pad_floor
                        (
                            &[
                                4.3, 4.4, 4.6, 4.9, 5.0, 5.2, 6.1, 6.2, 6.4, 7.9, 8.0, 8.2, 8.5,
                                8.6, 8.799999, 9.7, 9.799999, 10.0, 15.1, 15.2, 15.4, 15.7, 15.8,
                                16.0, 16.9, 17.0, 17.2,
                            ],
                            [1, 1, 3, 3, 3],
                        ),
                        // exclude_pad_ceil
                        (
                            &[
                                4.3, 4.4, 4.6, 4.7, 4.9, 5.0, 5.2, 5.3, 6.1, 6.2, 6.4, 6.5, 6.7,
                                6.8, 7.0, 7.1, 7.9, 8.0, 8.2, 8.299999, 8.5, 8.6, 8.799999, 8.9,
                                9.7, 9.799999, 10.0, 10.1, 10.3, 10.4, 10.6, 10.7, 15.1, 15.2,
                                15.4, 15.5, 15.7, 15.8, 16.0, 16.1, 16.9, 17.0, 17.2, 17.3, 17.5,
                                17.6, 17.8, 17.9, 18.7, 18.8, 19.0, 19.1, 19.3, 19.4, 19.6, 19.7,
                                20.5, 20.6, 20.8, 20.9, 21.1, 21.2, 21.4, 21.5,
                            ],
                            [1, 1, 4, 4, 4],
                        ),
                        // include_pad_floor
                        (
                            &[
                                0.5375, 1.1, 1.15, 1.225, 2.5, 2.6, 1.525, 3.1, 3.2, 1.975, 4.0,
                                4.1, 4.25, 8.6, 8.799999, 4.85, 9.799999, 10.0, 3.775, 7.6, 7.7,
                                7.85, 15.8, 16.0, 8.45, 17.0, 17.2,
                            ],
                            [1, 1, 3, 3, 3],
                        ),
                        // include_pad_ceil
                        (
                            &[
                                0.5375, 1.1, 1.15, 1.175, 1.225, 2.5, 2.6, 2.65, 1.525, 3.1, 3.2,
                                3.25, 1.675, 3.4, 3.5, 3.55, 1.975, 4.0, 4.1, 4.15, 4.25, 8.6,
                                8.799999, 8.9, 4.85, 9.799999, 10.0, 10.1, 5.15, 10.4, 10.6, 10.7,
                                3.775, 7.6, 7.7, 7.75, 7.85, 15.8, 16.0, 16.1, 8.45, 17.0, 17.2,
                                17.3, 8.75, 17.6, 17.8, 17.9, 4.675, 9.4, 9.5, 9.55, 9.65, 19.4,
                                19.6, 19.7, 10.25, 20.6, 20.8, 20.9, 10.55, 21.2, 21.4, 21.5,
                            ],
                            [1, 1, 4, 4, 4],
                        ),
                        // same_upper
                        (
                            &[
                                4.3, 4.5, 4.6, 5.5, 5.7, 5.8, 6.1, 6.3, 6.4, 11.5, 11.7, 11.8,
                                12.7, 12.9, 13.0, 13.3, 13.5, 13.6, 15.1, 15.3, 15.4, 16.3, 16.5,
                                16.6, 16.9, 17.1, 17.2,
                            ],
                            [1, 1, 3, 3, 3],
                        ),
                        // same_lower
                        (
                            &[
                                4.3, 4.4, 4.6, 4.9, 5.0, 5.2, 6.1, 6.2, 6.4, 7.9, 8.0, 8.2, 8.5,
                                8.6, 8.799999, 9.7, 9.799999, 10.0, 15.1, 15.2, 15.4, 15.7, 15.8,
                                16.0, 16.9, 17.0, 17.2,
                            ],
                            [1, 1, 3, 3, 3],
                        ),
                    ],
                ),
                (
                    7,
                    [
                        // exclude_pad_floor
                        (
                            &[
                                5.7, 5.8, 6.0, 6.1, 6.4, 6.5, 6.7, 6.8, 7.8, 7.9, 8.1, 8.2, 8.5,
                                8.6, 8.799999, 8.9, 10.6, 10.7, 10.9, 11.0, 11.3, 11.4, 11.6, 11.7,
                                12.7, 12.8, 13.0, 13.1, 13.4, 13.5, 13.7, 13.8, 20.4, 20.5, 20.7,
                                20.8, 21.1, 21.2, 21.4, 21.5, 22.5, 22.6, 22.8, 22.9, 23.2, 23.3,
                                23.5, 23.6, 25.3, 25.4, 25.6, 25.7, 26.0, 26.1, 26.3, 26.4, 27.4,
                                27.5, 27.7, 27.8, 28.1, 28.2, 28.4, 28.5,
                            ],
                            [1, 1, 4, 4, 4],
                        ),
                        // exclude_pad_ceil
                        (
                            &[
                                5.7, 5.8, 6.0, 6.1, 6.4, 6.5, 6.7, 6.8, 7.8, 7.9, 8.1, 8.2, 8.5,
                                8.6, 8.799999, 8.9, 10.6, 10.7, 10.9, 11.0, 11.3, 11.4, 11.6, 11.7,
                                12.7, 12.8, 13.0, 13.1, 13.4, 13.5, 13.7, 13.8, 20.4, 20.5, 20.7,
                                20.8, 21.1, 21.2, 21.4, 21.5, 22.5, 22.6, 22.8, 22.9, 23.2, 23.3,
                                23.5, 23.6, 25.3, 25.4, 25.6, 25.7, 26.0, 26.1, 26.3, 26.4, 27.4,
                                27.5, 27.7, 27.8, 28.1, 28.2, 28.4, 28.5,
                            ],
                            [1, 1, 4, 4, 4],
                        ),
                        // include_pad_floor
                        (
                            &[
                                0.7125, 1.45, 1.5, 0.7625, 1.6, 3.25, 3.35, 1.7, 1.95, 3.95, 4.05,
                                2.05, 1.0625, 2.15, 2.2, 1.1125, 2.65, 5.35, 5.45, 2.75, 5.65,
                                11.4, 11.6, 5.849999, 6.349999, 12.8, 13.0, 6.55, 3.35, 6.75, 6.85,
                                3.45, 5.1, 10.25, 10.35, 5.2, 10.55, 21.2, 21.4, 10.75, 11.25,
                                22.6, 22.8, 11.45, 5.8, 11.65, 11.75, 5.9, 3.1625, 6.35, 6.4,
                                3.2125, 6.5, 13.05, 13.15, 6.6, 6.85, 13.75, 13.85, 6.95, 3.5125,
                                7.05, 7.1, 3.5625,
                            ],
                            [1, 1, 4, 4, 4],
                        ),
                        // include_pad_ceil
                        (
                            &[
                                0.7125, 1.45, 1.5, 0.7625, 1.6, 3.25, 3.35, 1.7, 1.95, 3.95, 4.05,
                                2.05, 1.0625, 2.15, 2.2, 1.1125, 2.65, 5.35, 5.45, 2.75, 5.65,
                                11.4, 11.6, 5.849999, 6.349999, 12.8, 13.0, 6.55, 3.35, 6.75, 6.85,
                                3.45, 5.1, 10.25, 10.35, 5.2, 10.55, 21.2, 21.4, 10.75, 11.25,
                                22.6, 22.8, 11.45, 5.8, 11.65, 11.75, 5.9, 3.1625, 6.35, 6.4,
                                3.2125, 6.5, 13.05, 13.15, 6.6, 6.85, 13.75, 13.85, 6.95, 3.5125,
                                7.05, 7.1, 3.5625,
                            ],
                            [1, 1, 4, 4, 4],
                        ),
                        // same_upper
                        (
                            &[
                                5.7, 5.8, 6.0, 6.1, 6.4, 6.5, 6.7, 6.8, 7.8, 7.9, 8.1, 8.2, 8.5,
                                8.6, 8.799999, 8.9, 10.6, 10.7, 10.9, 11.0, 11.3, 11.4, 11.6, 11.7,
                                12.7, 12.8, 13.0, 13.1, 13.4, 13.5, 13.7, 13.8, 20.4, 20.5, 20.7,
                                20.8, 21.1, 21.2, 21.4, 21.5, 22.5, 22.6, 22.8, 22.9, 23.2, 23.3,
                                23.5, 23.6, 25.3, 25.4, 25.6, 25.7, 26.0, 26.1, 26.3, 26.4, 27.4,
                                27.5, 27.7, 27.8, 28.1, 28.2, 28.4, 28.5,
                            ],
                            [1, 1, 4, 4, 4],
                        ),
                        // same_lower
                        (
                            &[
                                5.7, 5.8, 6.0, 6.1, 6.4, 6.5, 6.7, 6.8, 7.8, 7.9, 8.1, 8.2, 8.5,
                                8.6, 8.799999, 8.9, 10.6, 10.7, 10.9, 11.0, 11.3, 11.4, 11.6, 11.7,
                                12.7, 12.8, 13.0, 13.1, 13.4, 13.5, 13.7, 13.8, 20.4, 20.5, 20.7,
                                20.8, 21.1, 21.2, 21.4, 21.5, 22.5, 22.6, 22.8, 22.9, 23.2, 23.3,
                                23.5, 23.6, 25.3, 25.4, 25.6, 25.7, 26.0, 26.1, 26.3, 26.4, 27.4,
                                27.5, 27.7, 27.8, 28.1, 28.2, 28.4, 28.5,
                            ],
                            [1, 1, 4, 4, 4],
                        ),
                    ],
                ),
            ],
        );
    }

    #[test]
    fn avg_pool1d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: avg_pool1d::Model = avg_pool1d::Model::new(&device);

        // Run the model
        let input = Tensor::<3>::from_floats(
            [[
                [-1.526, -0.750, -0.654, -1.609, -0.100],
                [-0.609, -0.980, -1.609, -0.712, 1.171],
                [1.767, -0.095, 0.139, -1.579, -0.321],
                [-0.299, 1.879, 0.336, 0.275, 1.716],
                [-0.056, 0.911, -1.392, 2.689, -0.111],
            ]],
            &device,
        );
        let (output1, output2, output3) = model.forward(input);
        let expected1 = TensorData::from([[[-1.135f32], [-0.978], [0.058], [0.548], [0.538]]]);
        let expected2 = TensorData::from([[
            [-0.569f32, -1.135, -0.591],
            [-0.397, -0.978, -0.288],
            [0.418, 0.058, -0.440],
            [0.395, 0.548, 0.582],
            [0.214, 0.538, 0.296],
        ]]);
        let expected3 = TensorData::from([[
            [-1.138f32, -1.135, -0.788],
            [-0.794, -0.978, -0.383],
            [0.836, 0.058, -0.587],
            [0.790, 0.548, 0.776],
            [0.427, 0.538, 0.395],
        ]]);

        let expected_shape1 = Shape::from([1, 5, 1]);
        let expected_shape2 = Shape::from([1, 5, 3]);
        let expected_shape3 = Shape::from([1, 5, 3]);

        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);
        assert_eq!(output3.shape(), expected_shape3);

        let tolerance = Tolerance::default();
        output1
            .to_data()
            .assert_approx_eq::<f32>(&expected1, tolerance);
        output2
            .to_data()
            .assert_approx_eq::<f32>(&expected2, tolerance);
        output3
            .to_data()
            .assert_approx_eq::<f32>(&expected3, tolerance);
    }

    #[test]
    fn avg_pool2d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: avg_pool2d::Model = avg_pool2d::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats(
            [[[
                [-0.077, 0.360, -0.782, 0.072, 0.665],
                [-0.287, 1.621, -1.597, -0.052, 0.611],
                [0.760, -0.034, -0.345, 0.494, -0.078],
                [-1.805, -0.476, 0.205, 0.338, 1.353],
                [0.374, 0.013, 0.774, -0.109, -0.271],
            ]]],
            &device,
        );
        let (output1, output2, output3) = model.forward(input);
        let expected1 = TensorData::from([[[[0.008f32, -0.131, -0.208, 0.425]]]]);
        let expected2 = TensorData::from([[[
            [-0.045f32, 0.202, -0.050, -0.295, 0.162, 0.160],
            [-0.176, 0.008, -0.131, -0.208, 0.425, 0.319],
            [-0.084, -0.146, 0.017, 0.170, 0.216, 0.125],
        ]]]);
        let expected3 = TensorData::from([[[
            [-0.182f32, 0.404, -0.100, -0.590, 0.324, 0.638],
            [-0.352, 0.008, -0.131, -0.208, 0.425, 0.638],
            [-0.224, -0.195, 0.023, 0.226, 0.288, 0.335],
        ]]]);

        let expected_shape1 = Shape::from([1, 1, 1, 4]);
        let expected_shape2 = Shape::from([1, 1, 3, 6]);
        let expected_shape3 = Shape::from([1, 1, 3, 6]);

        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);
        assert_eq!(output3.shape(), expected_shape3);

        let tolerance = Tolerance::rel_abs(0.01, 0.001);
        output1
            .to_data()
            .assert_approx_eq::<f32>(&expected1, tolerance);
        output2
            .to_data()
            .assert_approx_eq::<f32>(&expected2, tolerance);
        output3
            .to_data()
            .assert_approx_eq::<f32>(&expected3, tolerance);
    }

    #[test]
    fn avg_pool1d_ceil_mode() {
        // Test ceil_mode=True for AvgPool1d
        // Input: 1x1x6 (values 0-5), kernel: 3, stride: 2, padding: 0
        // With ceil_mode=True: output = ceil((6-3)/2)+1 = 3 elements
        let device = Default::default();
        let model: avg_pool1d_ceil_mode::Model = avg_pool1d_ceil_mode::Model::new(&device);

        let input = Tensor::<3>::from_floats([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]], &device);
        let output = model.forward(input);

        // Window 0: avg(0,1,2) = 1
        // Window 1: avg(2,3,4) = 3
        // Window 2: avg(4,5) = 4.5 (partial window, count_include_pad=False)
        let expected = TensorData::from([[[1.0f32, 3.0, 4.5]]]);
        let tolerance = Tolerance::rel_abs(0.01, 0.001);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, tolerance);
    }

    #[test]
    fn avg_pool1d_ceil_mode_opset10() {
        // ceil_mode arrived in AveragePool-10, so it must import below opset 19.
        let device = Default::default();
        let model: avg_pool1d_ceil_mode_opset10::Model =
            avg_pool1d_ceil_mode_opset10::Model::from_file(
                concat!(env!("OUT_DIR"), "/model/avg_pool1d_ceil_mode_opset10.bpk"),
                &device,
            );

        let input = Tensor::<3>::from_floats([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]], &device);
        let output = model.forward(input);

        // Ground truth from onnx.reference.ReferenceEvaluator
        // (avg_pool1d_ceil_mode_opset10.py).
        let expected = TensorData::from([[[1.0f32, 3.0, 4.5]]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn avg_pool2d_ceil_mode() {
        // Test ceil_mode=True for AvgPool2d
        // Input: 1x1x6x6 (values 0-35), kernel: 3x3, stride: 2x2, padding: 0
        // With ceil_mode=True: output = 3x3
        let device = Default::default();
        let model: avg_pool2d_ceil_mode::Model = avg_pool2d_ceil_mode::Model::new(&device);

        let input = Tensor::<4>::from_floats(
            [[[
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
                [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
            ]]],
            &device,
        );
        let output = model.forward(input);

        // With ceil_mode=True and count_include_pad=False:
        // (0,0): avg of 9 values = 7
        // (0,1): avg of 9 values = 9
        // (0,2): avg of 6 values (partial window) = 10.5
        // (1,0): avg of 9 values = 19
        // (1,1): avg of 9 values = 21
        // (1,2): avg of 6 values = 22.5
        // (2,0): avg of 6 values = 28
        // (2,1): avg of 6 values = 30
        // (2,2): avg of 4 values = 31.5
        let expected =
            TensorData::from([[[[7.0f32, 9.0, 10.5], [19.0, 21.0, 22.5], [28.0, 30.0, 31.5]]]]);
        let tolerance = Tolerance::rel_abs(0.01, 0.001);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, tolerance);
    }

    #[test]
    fn avg_pool1d_asymmetric_padding() {
        // Test asymmetric padding (left=1, right=2) for AvgPool1d
        let device = Default::default();
        let model: avg_pool1d_asymmetric_padding::Model =
            avg_pool1d_asymmetric_padding::Model::new(&device);

        // Run the model with ones as input for easier testing
        let input = Tensor::<3>::ones([2, 4, 10], &device);
        let output = model.forward(input);

        // With asymmetric padding (1, 2), input length 10 becomes 10+1+2=13
        // After pool with kernel 3, stride 1, output length is 13-3+1=11
        let expected_shape = Shape::from([2, 4, 11]);
        assert_eq!(output.shape(), expected_shape);

        // Verify the sum matches PyTorch output
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum = 77.333_33; // from pytorch
        assert!(
            (output_sum - expected_sum).abs() < 0.1,
            "Expected sum ~{}, got {}",
            expected_sum,
            output_sum
        );
    }

    #[test]
    fn avg_pool2d_asymmetric_padding() {
        // Test asymmetric padding: top=1, left=1, bottom=2, right=2 (ONNX pads: [1,1,2,2])
        let device = Default::default();
        let model: avg_pool2d_asymmetric_padding::Model =
            avg_pool2d_asymmetric_padding::Model::new(&device);

        // Run the model with ones as input for easier testing
        let input = Tensor::<4>::ones([2, 4, 10, 15], &device);
        let output = model.forward(input);

        // With asymmetric padding (1, 1, 2, 2), input (10, 15) becomes (13, 18)
        // After pool with kernel (3, 3), stride (1, 1), output is (11, 16)
        let expected_shape = Shape::from([2, 4, 11, 16]);
        assert_eq!(output.shape(), expected_shape);

        // Verify the sum matches ReferenceEvaluator output
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum = 1134.222; // from ReferenceEvaluator
        assert!(
            (output_sum - expected_sum).abs() < 1.0,
            "Expected sum ~{}, got {}",
            expected_sum,
            output_sum
        );
    }

    #[test]
    fn avg_pool2d_same_upper_dynamic() {
        // auto_pad=SAME_UPPER over dynamic H/W. kernel 2 / stride 2 over an extent of 5 leaves
        // a total pad of 1 at the end, so the trailing windows are half padding.
        // count_include_pad=1 because burn ignores 0 under asymmetric padding
        // (tracel-ai/burn#5450); see the generator script.
        let device = Default::default();
        let model: avg_pool2d_same_upper_dynamic::Model =
            avg_pool2d_same_upper_dynamic::Model::from_file(
                concat!(env!("OUT_DIR"), "/model/avg_pool2d_same_upper_dynamic.bpk"),
                &device,
            );

        let input = Tensor::<1, burn::tensor::Int>::arange(1..26, &device)
            .reshape([1, 1, 5, 5])
            .float();
        let output = model.forward(input);

        // Ground truth from onnx.reference.ReferenceEvaluator
        // (avg_pool2d_same_upper_dynamic.py).
        let expected = TensorData::from([[[
            [4.0f32, 6.0, 3.75],
            [14.0, 16.0, 8.75],
            [10.75, 11.75, 6.25],
        ]]]);
        output.to_data().assert_eq(&expected, true);
    }
}
