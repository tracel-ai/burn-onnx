use crate::include_models;
include_models!(col2im_basic, col2im_complex);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use alloc::vec;
    use burn::tensor::{Int, Tensor, TensorData, Tolerance};

    #[test]
    fn test_col2im_basic() {
        let device = Default::default();
        let model = col2im_basic::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::from_data(
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
        let model = col2im_complex::Model::<TestBackend>::new(&device);

        // Input: [1, 9, 9] from col2im.py (N=1, C_in=9, L=9)
        // We use pattern input to verify index mapping (1..82)
        let input = Tensor::<TestBackend, 1, Int>::arange(1..82, &device)
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
}
