use crate::include_models;
include_models!(col2im_basic, col2im_complex);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn::tensor::{Tensor, TensorData, Tolerance};
    use crate::backend::TestBackend;

    #[test]
    fn test_col2im_basic() {
        let device = Default::default();
        let model = col2im_basic::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![
                1.,  5.,  9., 13.,
                2.,  6., 10., 14.,
                3.,  7., 11., 15.,
                4.,  8., 12., 16.
            ], [1, 4, 4]),
            &device
        );
        let output = model.forward(input);
        let expected = TensorData::new(vec![
            1.,  2.,  5.,  6.,
            3.,  4.,  7.,  8.,
            9., 10., 13., 14.,
            11., 12., 15., 16.
        ], [1, 1, 4, 4]);
        
        output.into_data().assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn test_col2im_complex() {
        let device = Default::default();
        let model = col2im_complex::Model::<TestBackend>::new(&device);
        // Input: [1, 9, 9] from col2im.py (N=1, C_in=9, L=9)
        // We use ones for simplicity.
        let input = Tensor::<TestBackend, 3>::ones([1, 9, 9], &device);
        let output = model.forward(input);
        
        // Expected Output Shape: [1, 1, 5, 5]
        // (Image 5x5, N=1, C_out=1)
        let dims = output.shape().dims();
        assert_eq!(dims, [1, 1, 5, 5]);
    }
}
