// Import the shared macro
use crate::include_models;
include_models!(global_avr_pool, global_avr_pool_3d, global_avr_pool_squeeze);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Shape, Tensor, TensorData, Tolerance};
    use float_cmp::ApproxEq;

    #[test]
    fn globalavrpool_1d_2d() {
        // The model contains 1d and 2d global average pooling nodes
        let model: global_avr_pool::Model = global_avr_pool::Model::default();

        let device = Default::default();
        // Run the model with ones as input for easier testing
        let input_1d = Tensor::<3>::ones([2, 4, 10], &device);
        let input_2d = Tensor::<4>::ones([3, 10, 3, 15], &device);

        let (output_1d, output_2d) = model.forward(input_1d, input_2d);

        let expected_shape_1d = Shape::from([2, 4, 1]);
        let expected_shape_2d = Shape::from([3, 10, 1, 1]);
        assert_eq!(output_1d.shape(), expected_shape_1d);
        assert_eq!(output_2d.shape(), expected_shape_2d);

        let output_sum_1d = output_1d.sum().into_scalar::<f32>();
        let output_sum_2d = output_2d.sum().into_scalar::<f32>();

        let expected_sum_1d = 8.0; // from pytorch
        let expected_sum_2d = 30.0; // from pytorch

        assert!(expected_sum_1d.approx_eq(output_sum_1d, (1.0e-4, 2)));
        assert!(expected_sum_2d.approx_eq(output_sum_2d, (1.0e-4, 2)));
    }

    /// Squeeze with no axes drops every size-1 dim, so it only reaches rank 2 when
    /// GlobalAveragePool reports its spatial dims as 1 in the output static shape.
    #[test]
    fn globalavrpool_then_squeeze() {
        let model: global_avr_pool_squeeze::Model = global_avr_pool_squeeze::Model::default();

        let device = Default::default();
        let input = Tensor::<1, burn::tensor::Int>::arange(0..120, &device)
            .float()
            .reshape([2, 3, 4, 5]);

        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([[9.5f32, 29.5, 49.5], [69.5, 89.5, 109.5]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    /// Rank 5 has no adaptive pool module and reduces the three spatial axes with
    /// `mean_dims` instead.
    #[test]
    fn globalavrpool_3d() {
        let model: global_avr_pool_3d::Model = global_avr_pool_3d::Model::default();

        let device = Default::default();
        let input = Tensor::<1, burn::tensor::Int>::arange(0..144, &device)
            .float()
            .reshape([2, 3, 2, 3, 4]);

        let output = model.forward(input);

        let expected = TensorData::from([
            [[[[11.5f32]]], [[[35.5]]], [[[59.5]]]],
            [[[[83.5]]], [[[107.5]]], [[[131.5]]]],
        ]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
