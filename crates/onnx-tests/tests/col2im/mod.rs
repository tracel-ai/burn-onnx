use crate::include_models;
include_models!(
    col2im_basic,
    col2im_complex,
    col2im_asym,
    col2im_1d,
    col2im_runtime
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn::tensor::{Device, Int, Tensor, TensorData, Tolerance};

    #[test]
    fn test_col2im_basic() {
        let device = Default::default();
        let model = col2im_basic::Model::new(&device);
        let input = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    1., 5., 9., 13., 2., 6., 10., 14., 3., 7., 11., 15., 4., 8., 12., 16.,
                ],
                [1, 4, 4],
            ),
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::new(
            vec![
                1., 2., 5., 6., 3., 4., 7., 8., 9., 10., 13., 14., 11., 12., 15., 16.,
            ],
            [1, 1, 4, 4],
        );

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn test_col2im_complex() {
        // Test with strides, padding, dilation
        let device = Default::default();
        let model = col2im_complex::Model::new(&device);

        // Input: [1, 9, 9] from col2im.py (N=1, C_in=9, L=9)
        // We use pattern input to verify index mapping (1..82)
        let input = Tensor::<1, Int>::arange(1..82, &device)
            .reshape([1, 9, 9])
            .float();

        let output = model.forward(input);

        // Expected Output Shape: [1, 1, 5, 5]
        let dims = output.shape().dims();
        assert_eq!(dims, [1, 1, 5, 5]);

        // Expected Output Values (from onnx.reference in col2im.py)
        let expected = TensorData::new(
            vec![
                37.0, 75.0, 38.0, 77.0, 39.0, 77.0, 156.0, 79.0, 160.0, 81.0, 40.0, 81.0, 41.0,
                83.0, 42.0, 83.0, 168.0, 85.0, 172.0, 87.0, 43.0, 87.0, 44.0, 89.0, 45.0,
            ],
            [1, 1, 5, 5],
        );

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn test_col2im_asymmetric_pads() {
        let device = Default::default();
        let model = col2im_asym::Model::new(&device);
        let input = burn::tensor::Tensor::<1, burn::tensor::Int>::arange(0..40, &device)
            .float()
            .reshape([1, 4, 10]);

        let output = model.forward(input);

        output.to_data().assert_eq(
            &burn::tensor::TensorData::from([[[
                [20.0f32, 21.0, 52.0, 54.0, 56.0],
                [5.0, 6.0, 22.0, 24.0, 26.0],
                [25.0, 26.0, 62.0, 64.0, 66.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]]]),
            true,
        );
    }

    #[test]
    fn test_col2im_1d() {
        let device = Default::default();
        let model = col2im_1d::Model::new(&device);
        let input = burn::tensor::Tensor::<1, burn::tensor::Int>::arange(0..24, &device)
            .float()
            .reshape([1, 6, 4]);

        let output = model.forward(input);

        output.to_data().assert_eq(
            &burn::tensor::TensorData::from([[
                [0.0f32, 5.0, 15.0, 18.0, 17.0, 11.0],
                [12.0, 29.0, 51.0, 54.0, 41.0, 23.0],
            ]]),
            true,
        );
    }

    #[test]
    fn test_col2im_runtime_image_shape() {
        // Expected values from col2im_runtime.py (onnx ReferenceEvaluator).
        let device = Default::default();
        let model = col2im_runtime::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/col2im_runtime.bpk"),
            &device,
        );
        let x2d = Tensor::<3>::from_data(
            TensorData::new(
                (0..48).map(|v| v as f32).collect::<alloc::vec::Vec<_>>(),
                [1, 4, 12],
            ),
            &device,
        );
        let x1d = Tensor::<3>::from_data(
            TensorData::new(
                (0..9).map(|v| v as f32).collect::<alloc::vec::Vec<_>>(),
                [1, 3, 3],
            ),
            &device,
        );
        let image2d = Tensor::<1, Int>::from_ints([3, 3], &device);
        let image1d = Tensor::<1, Int>::from_ints([5], &device);

        let (y2d, y1d) = model.forward(x2d, image2d, x1d, image1d);

        y2d.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[
                [28.0f32, 82.0, 86.0],
                [36.0, 98.0, 102.0],
                [32.0, 77.0, 79.0],
            ]]]),
            Tolerance::default(),
        );
        y1d.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[0.0f32, 4.0, 12.0, 12.0, 8.0]]]),
            Tolerance::default(),
        );
    }
}
