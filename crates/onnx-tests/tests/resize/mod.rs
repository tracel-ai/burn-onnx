use crate::include_models;
include_models!(
    resize_options,
    resize_1d_linear_scale,
    resize_1d_nearest_scale,
    resize_2d_bicubic_scale,
    resize_2d_bilinear_scale,
    resize_2d_bilinear_half_pixel,
    resize_2d_nearest_scale,
    resize_with_sizes,
    resize_with_shape,
    resize_with_sizes_tensor,
    resize_with_scales_tensor,
    resize_axes_static,
    resize_axes_runtime,
    resize_keep_aspect_ratio_static,
    resize_keep_aspect_ratio_runtime,
    resize_keep_aspect_ratio_halfway,
    resize_chained,
    resize_chained_unknown_shape
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn::tensor::{Device, Tensor, TensorData};
    use float_cmp::ApproxEq;

    #[test]
    fn resize_preserves_options() {
        let device = Default::default();
        let model = resize_options::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/resize_options.bpk"),
            &device,
        );
        let input = Tensor::<4>::from_floats([[[[10.0, 20.0, 30.0, 40.0]]]], &device);
        let scales = Tensor::<1>::from_floats([1.0, 1.0, 1.0, 0.6], &device);
        let (static_nearest, runtime_nearest, exact, static_cubic, runtime_cubic) =
            model.forward(input, scales);
        let expected = TensorData::from([[[[10.0f32, 20.0]]]]);
        static_nearest.to_data().assert_eq(&expected, true);
        runtime_nearest.to_data().assert_eq(&expected, true);
        exact
            .to_data()
            .assert_eq(&TensorData::from([[[[20.0f32, 40.0]]]]), true);
        // ReferenceEvaluator results from resize_options.py.
        let expected_cubic = TensorData::from([[[[12.592592f32, 30.0]]]]);
        static_cubic
            .to_data()
            .assert_approx_eq::<f32>(&expected_cubic, burn::tensor::Tolerance::default());
        runtime_cubic
            .to_data()
            .assert_approx_eq::<f32>(&expected_cubic, burn::tensor::Tolerance::default());
    }

    #[test]
    fn resize_with_sizes() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_with_sizes::Model = resize_with_sizes::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats(
            [[[
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ]]],
            &device,
        );

        // The sizes are [1, 1, 2, 3]
        let output = model.forward(input);
        let expected = TensorData::from([[[[0.0f32, 1.5, 3.0], [12.0, 13.5, 15.0]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn resize_with_shape() {
        // Initialize the model without weights
        let device = Default::default();
        let model: resize_with_shape::Model = resize_with_shape::Model::new(&device);

        // Create input tensor [1, 3, 4, 4]
        let input = Tensor::<4>::from_floats(
            [[
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ],
                [
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                    [24.0, 25.0, 26.0, 27.0],
                    [28.0, 29.0, 30.0, 31.0],
                ],
                [
                    [32.0, 33.0, 34.0, 35.0],
                    [36.0, 37.0, 38.0, 39.0],
                    [40.0, 41.0, 42.0, 43.0],
                    [44.0, 45.0, 46.0, 47.0],
                ],
            ]],
            &device,
        );

        // The model should resize from [1, 3, 4, 4] to [1, 3, 8, 8] using bilinear interpolation
        let output = model.forward(input);

        // Check output dimensions
        assert_eq!(output.dims(), [1, 3, 8, 8]);

        // Check that interpolation is working correctly by verifying corner values
        // The corners should match the original corners
        let output_data = output.to_data();
        let values: Vec<f32> = output_data.try_into_vec().unwrap();

        // Check first channel corners
        assert_eq!(values[0], 0.0); // Top-left of first channel
        assert_eq!(values[7], 3.0); // Top-right of first channel
        assert_eq!(values[56], 12.0); // Bottom-left of first channel
        assert_eq!(values[63], 15.0); // Bottom-right of first channel

        // Check that the output has the right number of elements
        assert_eq!(values.len(), 1 * 3 * 8 * 8);
    }

    #[test]
    fn resize_with_sizes_tensor() {
        // Initialize the model without weights
        let device = Default::default();
        let model: resize_with_sizes_tensor::Model = resize_with_sizes_tensor::Model::new(&device);

        // Create input tensor [1, 3, 4, 4]
        let input = Tensor::<4>::from_floats(
            [[
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0],
                    [29.0, 30.0, 31.0, 32.0],
                ],
                [
                    [33.0, 34.0, 35.0, 36.0],
                    [37.0, 38.0, 39.0, 40.0],
                    [41.0, 42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0, 48.0],
                ],
            ]],
            &device,
        );

        // Create sizes tensor [1, 3, 2, 2] - resize to 2x2
        let sizes = Tensor::<1, burn::tensor::Int>::from_ints([1i64, 3, 2, 2], &device);

        // The model should resize from [1, 3, 4, 4] to [1, 3, 2, 2] using nearest neighbor
        let output = model.forward(input, sizes);

        // Check output dimensions
        assert_eq!(output.dims(), [1, 3, 2, 2]);

        // With nearest neighbor and downsampling by 2, we should get corners
        let output_data = output.to_data();
        let values: Vec<f32> = output_data.try_into_vec().unwrap();

        // For nearest neighbor with 2x downsampling, we expect to sample at positions (0,0), (0,2), (2,0), (2,2)
        // First channel should have values [1, 3, 9, 11]
        assert_eq!(values[0], 1.0); // [0, 0, 0, 0]
        assert_eq!(values[1], 3.0); // [0, 0, 0, 2]
        assert_eq!(values[2], 9.0); // [0, 0, 2, 0]
        assert_eq!(values[3], 11.0); // [0, 0, 2, 2]

        // Check that the output has the right number of elements
        assert_eq!(values.len(), 1 * 3 * 2 * 2);
    }

    #[test]
    fn resize_with_scales_1d_linear() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_1d_linear_scale::Model = resize_1d_linear_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<3>::from_floats(
            [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
            &device,
        );

        // The scales are 1.5
        let output = model.forward(input);

        Tensor::<3>::from([[[
            1.5410, 0.3945, -0.7648, -1.9431, -0.8052, 0.3618, -0.6713, -1.2023, -1.3986,
        ]]])
        .to_data()
        .assert_approx_eq::<f32>(&output.into_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn resize_with_scales_2d_bilinear_half_pixel() {
        let device = Default::default();
        let model: resize_2d_bilinear_half_pixel::Model =
            resize_2d_bilinear_half_pixel::Model::new(&device);

        let input = Tensor::<4>::from_floats(
            [[[
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ]]],
            &device,
        );

        // Scales [1, 1, 2, 2] with coordinate_transformation_mode="half_pixel"
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 1, 8, 8]);

        // Reference from ONNX ReferenceEvaluator: sum = 480.0
        let output_data = output.to_data();
        let values: Vec<f32> = output_data.try_into_vec().unwrap();
        let output_sum: f32 = values.iter().sum();
        assert!(480.0f32.approx_eq(output_sum, (1.0e-4, 2)));

        // Verify half_pixel behavior: middle values differ from align_corners
        assert!(values[0].approx_eq(0.0, (1.0e-4, 2)));
        assert!(values[1].approx_eq(0.25, (1.0e-2, 2)));
    }

    #[test]
    fn resize_with_scales_2d_bilinear() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_bilinear_scale::Model = resize_2d_bilinear_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum = -3.401_126_6; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_2d_nearest() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_nearest_scale::Model = resize_2d_nearest_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9, 9]);

        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum = -0.812_227_7; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_1d_nearest() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_1d_nearest_scale::Model = resize_1d_nearest_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<3>::from_floats(
            [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9]);

        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum = -4.568_224; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_2d_bicubic() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_bicubic_scale::Model = resize_2d_bicubic_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9, 9]);

        let output_sum = output.sum().into_scalar::<f32>();

        let expected_sum = -3.515_921; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-3, 2)));
    }

    #[test]
    fn resize_with_scales_tensor() {
        // Initialize the model without weights
        let device = Default::default();
        let model: resize_with_scales_tensor::Model =
            resize_with_scales_tensor::Model::new(&device);

        // Create input tensor [1, 3, 4, 4]
        let input = Tensor::<4>::from_floats(
            [[
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0],
                    [29.0, 30.0, 31.0, 32.0],
                ],
                [
                    [33.0, 34.0, 35.0, 36.0],
                    [37.0, 38.0, 39.0, 40.0],
                    [41.0, 42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0, 48.0],
                ],
            ]],
            &device,
        );

        // Create scales tensor to double spatial dimensions: [1, 3, 4, 4] -> [1, 3, 8, 8]
        // Format: [scale_n, scale_c, scale_h, scale_w]
        let scales = Tensor::<1>::from_floats([1.0f32, 1.0, 2.0, 2.0], &device);

        // The model should resize from [1, 3, 4, 4] to [1, 3, 8, 8] using nearest neighbor
        let output = model.forward(input, scales);

        // Check output dimensions
        assert_eq!(output.dims(), [1, 3, 8, 8]);

        // With nearest neighbor and 2x upsampling, output sum should be 4x input sum
        // Input sum per channel: 136, 392, 648 => total = 1176
        // Output sum should be 4704 (each value repeated 4 times)
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum = 4704.0f32;

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_axes_static() {
        // Expected values from resize_axes_static.py (onnx ReferenceEvaluator).
        // axes=[3] with constant sizes [6]: only the width is resized.
        let device = Default::default();
        let model = resize_axes_static::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/resize_axes_static.bpk"),
            &device,
        );
        let input = Tensor::<4>::from_floats([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]], &device);

        let output = model.forward(input);

        output.to_data().assert_eq(
            &TensorData::from([[[
                [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            ]]]),
            true,
        );
    }

    #[test]
    fn resize_axes_runtime() {
        // Expected values from resize_axes_runtime.py (onnx ReferenceEvaluator).
        // axes=[3, 2] with runtime scales [3.0, 2.0]: width x3, height x2.
        let device = Default::default();
        let model = resize_axes_runtime::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/resize_axes_runtime.bpk"),
            &device,
        );
        let input = Tensor::<4>::from_floats([[[[0.0, 1.0], [2.0, 3.0]]]], &device);
        let scales = Tensor::<1>::from_floats([3.0, 2.0], &device);

        let output = model.forward(input, scales);

        output.to_data().assert_eq(
            &TensorData::from([[[
                [0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
            ]]]),
            true,
        );
    }

    #[test]
    fn resize_keep_aspect_ratio_static() {
        // Expected values from resize_keep_aspect_ratio_static.py (ONNX Runtime). Constant sizes
        // [7, 8] on a 2x2 input: not_larger -> 7x7, not_smaller -> 8x8 (adjusted at import).
        let device = Default::default();
        let model = resize_keep_aspect_ratio_static::Model::from_file(
            concat!(
                env!("OUT_DIR"),
                "/model/resize_keep_aspect_ratio_static.bpk"
            ),
            &device,
        );
        let input = Tensor::<4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);

        let (not_larger, not_smaller) = model.forward(input);

        // Rows/columns 0..4 come from the first input row/column, the rest from the second
        let expected = |n: usize, split: usize| {
            let mut values = Vec::with_capacity(n * n);
            for row in 0..n {
                for col in 0..n {
                    let top = if row < split { 0.0f32 } else { 2.0 };
                    let left = if col < split { 1.0 } else { 2.0 };
                    values.push(top + left);
                }
            }
            TensorData::new(values, [1, 1, n, n])
        };
        not_larger.to_data().assert_eq(&expected(7, 4), true);
        not_smaller.to_data().assert_eq(&expected(8, 4), true);
    }

    #[test]
    fn resize_keep_aspect_ratio_runtime_keeps_batch_and_channel() {
        // Expected values from resize_keep_aspect_ratio_runtime.py (ONNX Runtime). Without axes
        // every axis is listed: sizes [1, 1, 1, 2] -> scale max(1, 1, 0.5, 1) = 1, size unchanged.
        let device = Default::default();
        let model = resize_keep_aspect_ratio_runtime::Model::from_file(
            concat!(
                env!("OUT_DIR"),
                "/model/resize_keep_aspect_ratio_runtime.bpk"
            ),
            &device,
        );
        let input = Tensor::<4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
        let sizes = Tensor::<1, burn::tensor::Int>::from_ints([1, 1, 1, 2], &device);

        let output = model.forward(input, sizes);

        output
            .to_data()
            .assert_eq(&TensorData::from([[[[1.0f32, 2.0], [3.0, 4.0]]]]), true);
    }

    #[test]
    #[should_panic(expected = "Resize: keep_aspect_ratio_policy would resize axis 0")]
    fn resize_keep_aspect_ratio_runtime_rejects_batch_resize() {
        // sizes [1, 1, 2, 3] -> scale 1.5, which would resize the batch axis (ONNX Runtime
        // outputs 2x2x3x3); burn only resizes spatial axes, so this stops with a clear message.
        let device = Default::default();
        let model = resize_keep_aspect_ratio_runtime::Model::from_file(
            concat!(
                env!("OUT_DIR"),
                "/model/resize_keep_aspect_ratio_runtime.bpk"
            ),
            &device,
        );
        let input = Tensor::<4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
        let sizes = Tensor::<1, burn::tensor::Int>::from_ints([1, 1, 2, 3], &device);

        let _ = model.forward(input, sizes);
    }

    #[test]
    fn resize_keep_aspect_ratio_rounds_halfway_up() {
        // Expected shapes from resize_keep_aspect_ratio_halfway.py (ONNX Runtime). not_larger
        // with sizes [15, 29] on 25x50: scale 0.58 and 0.58 * 25 = 14.5 -> 15, with constant
        // sizes (adjusted at import) and runtime sizes (adjusted in forward).
        let device = Default::default();
        let model = resize_keep_aspect_ratio_halfway::Model::from_file(
            concat!(
                env!("OUT_DIR"),
                "/model/resize_keep_aspect_ratio_halfway.bpk"
            ),
            &device,
        );
        let input = Tensor::<4>::zeros([1, 1, 25, 50], &device);
        let sizes = Tensor::<1, burn::tensor::Int>::from_ints([15, 29], &device);

        let (y_static, y_runtime) = model.forward(input, sizes);

        assert_eq!(y_static.dims(), [1, 1, 15, 29]);
        assert_eq!(y_runtime.dims(), [1, 1, 15, 29]);
    }

    #[test]
    fn resize_chained_reads_the_resized_shape() {
        // Expected values from resize_chained.py (ONNX Runtime). A stretch Resize turns 2x4
        // into 4x2; the following Resizes must use 4x2, not the original 2x4 (3x6 and 2x5).
        let device = Default::default();
        let model = resize_chained::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/resize_chained.bpk"),
            &device,
        );
        let input =
            Tensor::<4>::from_floats([[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]], &device);

        let (y_not_larger, y_axes) = model.forward(input);

        y_not_larger.to_data().assert_eq(
            &TensorData::from([[[
                [0.0f32, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [4.0, 4.0, 6.0],
                [4.0, 4.0, 6.0],
                [4.0, 4.0, 6.0],
            ]]]),
            true,
        );
        y_axes.to_data().assert_eq(
            &TensorData::from([[[
                [0.0f32, 0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 2.0, 2.0],
                [4.0, 4.0, 4.0, 6.0, 6.0],
                [4.0, 4.0, 4.0, 6.0, 6.0],
            ]]]),
            true,
        );
    }

    #[test]
    fn resize_chained_after_unknown_shape() {
        // Expected values from resize_chained_unknown_shape.py (ONNX Runtime). GridSample's
        // output has no static shape; the stretch Resize's constant sizes [1, 1, 4, 2] must
        // still give the following not_larger Resize its 4x2 input (6x6 -> 6x3).
        let device = Default::default();
        let model = resize_chained_unknown_shape::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/resize_chained_unknown_shape.bpk"),
            &device,
        );
        let input =
            Tensor::<4>::from_floats([[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]], &device);

        let output = model.forward(input);

        output.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[
                [0.0f32, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [4.0, 4.0, 6.0],
                [4.0, 4.0, 6.0],
                [4.0, 4.0, 6.0],
            ]]]),
            burn::tensor::Tolerance::default(),
        );
    }
}
