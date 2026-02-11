use crate::include_models;
include_models!(deform_conv, deform_conv_bias, deform_conv_mask);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn deform_conv() {
        let device = Default::default();
        let model: deform_conv::Model<TestBackend> = deform_conv::Model::default();

        // X=[1,1,3,3] all ones, offset=[1,8,2,2] all zeros
        let input = Tensor::<TestBackend, 4>::ones([1, 1, 3, 3], &device);
        let offset = Tensor::<TestBackend, 4>::zeros([1, 8, 2, 2], &device);

        let output = model.forward(input, offset);

        let expected_shape = Shape::from([1, 1, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();
        let expected_sum = 10.116_673; // from ReferenceEvaluator
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn deform_conv_with_bias() {
        let device = Default::default();
        let model: deform_conv_bias::Model<TestBackend> = deform_conv_bias::Model::default();

        let input = Tensor::<TestBackend, 4>::ones([1, 1, 3, 3], &device);
        let offset = Tensor::<TestBackend, 4>::zeros([1, 8, 2, 2], &device);

        let output = model.forward(input, offset);

        let expected_shape = Shape::from([1, 1, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();
        let expected_sum = 12.116_673; // base (10.116673) + bias (0.5 * 4 elements)
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn deform_conv_with_mask() {
        let device = Default::default();
        let model: deform_conv_mask::Model<TestBackend> = deform_conv_mask::Model::default();

        let input = Tensor::<TestBackend, 4>::ones([1, 1, 3, 3], &device);
        let offset = Tensor::<TestBackend, 4>::zeros([1, 8, 2, 2], &device);
        let mask = Tensor::<TestBackend, 4>::ones([1, 4, 2, 2], &device);

        let output = model.forward(input, offset, mask);

        let expected_shape = Shape::from([1, 1, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();
        // mask=ones is same as no mask, so same as bias result
        let expected_sum = 12.116_673;
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }
}
