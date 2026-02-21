// Import the shared macro
use crate::include_models;
include_models!(global_max_pool);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn globalmaxpool_3d_4d() {
        // The model contains global max pooling nodes one with 3d input and another with 4d input.
        let model: global_max_pool::Model<TestBackend> = global_max_pool::Model::default();

        let device = Default::default();

        // Input 1: [2, 3, 4, 5] - 2 batches, 3 channels, 4x5 spatial
        let input1 = Tensor::<TestBackend, 4>::arange(0.0, 2.0 * 3.0 * 4.0 * 5.0, 1.0, &device)
            .reshape([2, 3, 4, 5]);

        // Input 2: [3, 10, 3, 15] - 3 batches, 10 channels, 3x15 spatial
        let input2 = Tensor::<TestBackend, 4>::arange(0.0, 3.0 * 10.0 * 3.0 * 15.0, 1.0, &device)
            .reshape([3, 10, 3, 15]);

        let (output1, output2) = model.forward(input1, input2);

        // Check output shapes
        let expected_shape1 = Shape::from([2, 3, 1, 1]);
        let expected_shape2 = Shape::from([3, 10, 1, 1]);
        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);

        // For input1 [2, 3, 4, 5]: each channel has 20 elements (4×5)
        // Batch 0: channels with values 0-19, 20-39, 40-59 → max: 19, 39, 59
        // Batch 1: channels with values 60-79, 80-99, 100-119 → max: 79, 99, 119
        let expected_output1 =
            Tensor::<TestBackend, 4>::from_slice(&[19.0, 39.0, 59.0, 79.0, 99.0, 119.0], &device)
                .reshape(expected_shape1);

        // For input2 [3, 10, 3, 15]: each channel has 45 elements (3×15)
        // Each channel max: batch*450 + channel*45 + 44
        let expected_values_2: Vec<f32> = (0..3)
            .flat_map(|b| (0..10).map(move |c| (b * 10 * 3 * 15 + c * 3 * 15 + 3 * 15 - 1) as f32))
            .collect();
        let expected_output2 = Tensor::<TestBackend, 4>::from_slice(&expected_values_2, &device)
            .reshape(expected_shape2);

        assert!(expected_output1.approx_eq(&output1, (1.0e-4, 2)));
        assert!(expected_output2.approx_eq(&output2, (1.0e-4, 2)));
    }
}
