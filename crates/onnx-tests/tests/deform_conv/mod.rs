use crate::include_models;
include_models!(
    deform_conv,
    deform_conv_bias,
    deform_conv_mask,
    deform_conv_runtime_bias
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Shape, Tensor};
    use float_cmp::ApproxEq;

    #[test]
    fn deform_conv() {
        let device = Default::default();
        let model: deform_conv::Model = deform_conv::Model::default();

        // X=[1,1,3,3] all ones, offset=[1,8,2,2] all zeros
        let input = Tensor::<4>::ones([1, 1, 3, 3], &device);
        let offset = Tensor::<4>::zeros([1, 8, 2, 2], &device);

        let output = model.forward(input, offset);

        let expected_shape = Shape::from([1, 1, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum: f32 = 10.116_673; // from ReferenceEvaluator
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn deform_conv_with_bias() {
        let device = Default::default();
        let model: deform_conv_bias::Model = deform_conv_bias::Model::default();

        let input = Tensor::<4>::ones([1, 1, 3, 3], &device);
        let offset = Tensor::<4>::zeros([1, 8, 2, 2], &device);

        let output = model.forward(input, offset);

        let expected_shape = Shape::from([1, 1, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum: f32 = 12.116_673; // base (10.116673) + bias (0.5 * 4 elements)
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn deform_conv_with_mask() {
        let device = Default::default();
        let model: deform_conv_mask::Model = deform_conv_mask::Model::default();

        let input = Tensor::<4>::ones([1, 1, 3, 3], &device);
        let offset = Tensor::<4>::zeros([1, 8, 2, 2], &device);
        let mask = Tensor::<4>::ones([1, 4, 2, 2], &device);

        let output = model.forward(input, offset, mask);

        let expected_shape = Shape::from([1, 1, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar::<f32>();
        // mask=ones is same as no mask, so same as bias result
        let expected_sum: f32 = 12.116_673;
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn deform_conv_runtime_bias() {
        // Constant weight with a runtime bias: both stay runtime so the bias is applied.
        let device = Default::default();
        let model = deform_conv_runtime_bias::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/deform_conv_runtime_bias.bpk"),
            &device,
        );

        let x = Tensor::<4>::from_floats([[[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]]], &device);
        let offset = Tensor::<4>::zeros([1, 8, 2, 2], &device);
        let bias = Tensor::<1>::from_floats([0.5], &device);

        let output = model.forward(x, offset, bias);
        let expected = burn::tensor::TensorData::from([[[[27.5f32, 37.5], [57.5, 67.5]]]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
