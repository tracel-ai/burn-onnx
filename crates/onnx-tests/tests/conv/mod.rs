// Import the shared macro
use crate::include_models;
include_models!(
    conv1d,
    conv1d_asymmetric_padding,
    conv1d_same_upper_dynamic,
    conv2d,
    conv2d_asymmetric_padding,
    conv2d_same_upper_dynamic,
    conv3d,
    conv3d_same,
    conv3d_asymmetric_padding
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Int, Shape, Tensor, TensorData, Tolerance};
    use core::f64::consts;
    use float_cmp::ApproxEq;

    #[test]
    fn conv3d_same_stride_two() {
        let device = Default::default();
        let model = conv3d_same::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/conv3d_same.bpk"),
            &device,
        );
        let values: alloc::vec::Vec<f32> = (0..120).map(|n| n as f32 / 10.0).collect();
        let input = Tensor::<5>::from_data(
            TensorData::new(values, [1, 1, 4, 5, 6]),
            (&device, burn::tensor::DType::F32),
        );
        let (upper, lower) = model.forward(input);
        for (actual, bytes) in [
            (upper, &include_bytes!("conv3d_same_0_upper.bin")[..]),
            (lower, &include_bytes!("conv3d_same_0_lower.bin")[..]),
        ] {
            let expected: alloc::vec::Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            actual.to_data().assert_approx_eq::<f32>(
                &TensorData::new(expected, [1, 1, 2, 3, 3]),
                Tolerance::absolute(1e-4),
            );
        }
    }

    #[test]
    fn conv1d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv1d::Model = conv1d::Model::default();

        // Run the model with pi as input for easier testing
        let input = Tensor::<3>::full([6, 4, 10], consts::PI, &Default::default());

        let output = model.forward(input);

        // test the output shape
        let expected_shape: Shape = Shape::from([6, 2, 7]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv1d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum: f32 = -54.549_243; // from pytorch
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv2d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv2d::Model = conv2d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 6, 15]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv2d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar::<f32>();

        // PyTorch f32 ground truth (from conv2d/conv2d.py, torch 2.10.0 CPU).
        // Tolerance accommodates gemm-order accumulation drift on the ~1080-element
        // output sum; burn-flex's gemm-backed conv stays within ~2e-4 of PyTorch.
        let expected_sum = -113.86999_5_f32;
        assert!(expected_sum.approx_eq(output_sum, (1.0e-3, 2)));
    }

    #[test]
    fn conv3d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv3d::Model = conv3d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<5>::ones([2, 4, 4, 5, 7], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 3, 5, 5]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv3d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar::<f32>();

        let expected_sum: f32 = 48.494_262; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv1d_asymmetric_padding() {
        // Initialize the model with weights (loaded from the exported file)
        // This model tests asymmetric padding: (left=1, right=2)
        let model: conv1d_asymmetric_padding::Model = conv1d_asymmetric_padding::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<3>::ones([2, 4, 10], &Default::default());

        let output = model.forward(input);

        // With asymmetric padding (1, 2), input length 10 becomes 10+1+2=13
        // After conv with kernel 3, stride 1, output length is 13-3+1=11
        let expected_shape = Shape::from([2, 6, 11]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum: f32 = -0.386_136; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-3, 2)));
    }

    #[test]
    fn conv2d_asymmetric_padding() {
        // Initialize the model with weights (loaded from the exported file)
        // This model tests asymmetric padding: (left=1, right=2, top=1, bottom=3)
        let model: conv2d_asymmetric_padding::Model = conv2d_asymmetric_padding::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        // With asymmetric padding (1, 2, 1, 3), input (10, 15) becomes (10+1+3, 15+1+2) = (14, 18)
        // After conv with kernel (3, 3), stride (1, 1), output is (12, 16)
        let expected_shape = Shape::from([2, 6, 12, 16]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar::<f32>();

        // PyTorch f32 ground truth (from conv/conv2d_asymmetric_padding.py, torch 2.10.0
        // CPU). Tolerance is a bit larger than conv2d because the ~2304-element output
        // sum magnifies gemm-order accumulation drift (~2.5e-3 absolute vs PyTorch).
        let expected_sum = -481.67495_7_f32;
        assert!(expected_sum.approx_eq(output_sum, (5.0e-3, 2)));
    }

    #[test]
    fn conv1d_same_upper_dynamic() {
        // The 1D counterpart of conv2d_same_upper_dynamic. Its job is to compile and run
        // PaddingConfig1d::Same, which the snapshot tests only assert as generated text.
        // stride 2 makes the pads depend on the parity of the length: L=9 -> (1, 2), L=8 -> (1, 1).
        let device = Default::default();
        let model: conv1d_same_upper_dynamic::Model = conv1d_same_upper_dynamic::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/conv1d_same_upper_dynamic.bpk"),
            &device,
        );

        // Ground truth from onnxruntime (conv1d_same_upper_dynamic.py).
        for (length, out_len, expected_sum) in
            [(9usize, 5usize, -23.750_193_f32), (8, 4, -19.424_349)]
        {
            let input = Tensor::<3>::ones([1, 2, length], &device);
            let output = model.forward(input);

            assert_eq!(output.shape(), Shape::from([1, 3, out_len]));

            let output_sum = output.sum().into_scalar::<f32>();
            assert!(
                expected_sum.approx_eq(output_sum, (1.0e-3, 2)),
                "L={length}: expected {expected_sum}, got {output_sum}"
            );
        }
    }

    #[test]
    fn conv2d_same_upper_dynamic() {
        // auto_pad=SAME_UPPER over dynamic H/W: burn derives the pads from the real input size
        // at forward time. stride 2 makes the pads depend on the parity of the extent, so the
        // two sizes below exercise different pads (5x6 -> H (1,1) W (1,1); 8x7 -> H (0,1)
        // W (1,2)), and the odd totals exercise SAME_UPPER putting the extra pad last.
        let device = Default::default();
        let model: conv2d_same_upper_dynamic::Model = conv2d_same_upper_dynamic::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/conv2d_same_upper_dynamic.bpk"),
            &device,
        );

        // Ground truth from onnxruntime (see conv2d_same_upper_dynamic.py for why not
        // ReferenceEvaluator).
        for (height, width, out_h, out_w, expected_sum) in [
            (5usize, 6usize, 3usize, 3usize, -45.890_423_f32),
            (8, 7, 4, 4, -123.938_919),
        ] {
            let input = Tensor::<4>::ones([1, 2, height, width], &device);
            let output = model.forward(input);

            assert_eq!(output.shape(), Shape::from([1, 3, out_h, out_w]));

            let output_sum = output.sum().into_scalar::<f32>();
            assert!(
                expected_sum.approx_eq(output_sum, (1.0e-3, 2)),
                "{height}x{width}: expected {expected_sum}, got {output_sum}"
            );
        }
    }

    #[test]
    fn conv3d_asymmetric_padding() {
        // Pads whose two ends differ on an axis, which burn-nn's symmetric
        // PaddingConfig3d cannot express (issue #646). y1-y3 have constant weights and
        // used to panic in codegen; y4 takes its weight at run time.
        let device = Default::default();
        let model = conv3d_asymmetric_padding::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/conv3d_asymmetric_padding.bpk"),
            &device,
        );

        let seq = |shape: [usize; 5], scale: f32| -> Tensor<5> {
            let n = shape.iter().product::<usize>() as i64;
            Tensor::<1, Int>::arange(0..n, &device)
                .float()
                .reshape(shape)
                .mul_scalar(scale)
                .sub_scalar(1.0)
        };
        let (y1, y2, y3, y4) = model.forward(seq([1, 2, 3, 3, 3], 0.1), seq([2, 2, 2, 2, 2], 0.04));

        // Ground truth from ReferenceEvaluator (conv/conv3d_asymmetric_padding.py).
        let tolerance = Tolerance::absolute(1e-4);
        // Constant weight, pads [0, 1, 0, 1, 0, 1].
        y1.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [
                    [
                        [-1.38f32, -1.84, -1.14],
                        [-4.82, -5.82, -3.4],
                        [-7.82, -8.82, -4.96],
                    ],
                    [
                        [-5.52, -5.98, -3.3],
                        [-13.82, -14.82, -8.08],
                        [-16.82, -17.82, -9.64],
                    ],
                    [
                        [-4.855, -5.125, -2.75],
                        [-11.4, -11.98, -6.39],
                        [-13.14, -13.72, -7.29],
                    ],
                ],
                [
                    [[4.06, 4.24, 2.06], [7.98, 8.26, 3.96], [8.82, 9.1, 4.32]],
                    [[5.68, 5.86, 2.78], [10.5, 10.78, 5.04], [11.34, 11.62, 5.4]],
                    [[2.185, 2.235, 1.01], [3.64, 3.7, 1.61], [3.82, 3.88, 1.67]],
                ],
            ]]),
            tolerance,
        );
        // Constant weight and bias, pads [1, 0, 2, 0, 1, 1], strides [2, 1, 1].
        y2.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [[
                    [0.752f32, 0.308, -0.856, -1.18],
                    [-0.124, -1.26, -2.924, -2.348],
                ]],
                [[
                    [13.508, 28.616, 44.3, 29.72],
                    [9.176, 19.704, 30.568, 20.344],
                ]],
            ]]),
            tolerance,
        );
        // Constant weight, SAME_UPPER over kernel [2, 3, 2]: pads (0, 1), (1, 1), (0, 1).
        y3.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [
                    [
                        [-5.38f32, -6.38, -3.584],
                        [-10.968, -12.54, -6.906],
                        [-9.628, -10.724, -5.816],
                    ],
                    [
                        [-14.38, -15.38, -8.192],
                        [-25.116, -26.688, -14.142],
                        [-19.492, -20.588, -10.856],
                    ],
                    [
                        [-11.456, -12.028, -6.31],
                        [-19.281, -20.175, -10.554],
                        [-14.444, -15.064, -7.858],
                    ],
                ],
                [
                    [
                        [6.14, 6.292, 3.04],
                        [8.904, 9.06, 4.326],
                        [5.348, 5.404, 2.536],
                    ],
                    [
                        [7.508, 7.66, 3.616],
                        [10.308, 10.464, 4.866],
                        [5.852, 5.908, 2.68],
                    ],
                    [
                        [2.08, 2.084, 0.89],
                        [2.319, 2.289, 0.894],
                        [0.82, 0.776, 0.206],
                    ],
                ],
            ]]),
            tolerance,
        );
        // Weight fed at run time, pads [1, 0, 1, 0, 1, 1].
        y4.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [
                    [
                        [-0.312f32, -0.944, -1.44, -0.904],
                        [-1.032, -2.432, -2.928, -1.672],
                        [-0.784, -1.772, -2.036, -1.132],
                    ],
                    [
                        [-3.104, -7.056, -8.176, -4.56],
                        [-4.736, -10.416, -11.536, -6.288],
                        [-3.024, -6.56, -7.152, -3.856],
                    ],
                    [
                        [-8.0, -17.136, -18.256, -9.744],
                        [-9.632, -20.496, -21.616, -11.472],
                        [-5.616, -11.888, -12.48, -6.592],
                    ],
                ],
                [
                    [
                        [0.968, 1.872, 1.888, 0.888],
                        [1.016, 1.92, 1.936, 0.888],
                        [0.432, 0.788, 0.78, 0.34],
                    ],
                    [
                        [1.76, 3.184, 3.088, 1.328],
                        [1.664, 2.896, 2.8, 1.136],
                        [0.56, 0.864, 0.784, 0.24],
                    ],
                    [
                        [1.472, 2.32, 2.224, 0.752],
                        [1.376, 2.032, 1.936, 0.56],
                        [0.272, 0.144, 0.064, -0.192],
                    ],
                ],
            ]]),
            tolerance,
        );
    }
}
