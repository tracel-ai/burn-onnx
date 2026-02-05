// Import the shared macro
use crate::include_models;
include_models!(batch_norm, batch_norm_runtime, batch_norm_partial_constant);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn batch_norm() {
        let model: batch_norm::Model<TestBackend> = batch_norm::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 3>::ones([1, 20, 1], &Default::default());
        let output = model.forward(input);

        let expected_shape = Shape::from([1, 5, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();
        let expected_sum = 19.999_802; // from pytorch
        assert!(expected_sum.approx_eq(output_sum, (1.0e-8, 2)));
    }

    #[test]
    fn batch_norm_runtime() {
        let model: batch_norm_runtime::Model<TestBackend> = batch_norm_runtime::Model::default();

        let device = Default::default();
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[
                [[0.4967, -0.1383], [0.6477, 1.5230]],
                [[-0.2342, -0.2341], [1.5792, 0.7674]],
                [[-0.4695, 0.5426], [-0.4634, -0.4657]],
            ]],
            &device,
        );
        let scale = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 0.5], &device);
        let bias = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0, -1.0], &device);
        let mean = Tensor::<TestBackend, 1>::from_floats([0.0, 0.5, -0.5], &device);
        let var = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 0.5], &device);

        let output = model.forward(input, scale, bias, mean, var);

        let expected_shape = Shape::from([1, 3, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        // Expected sum from ONNX ReferenceEvaluator: 3.166
        let output_sum = output.sum().into_scalar();
        assert!(
            3.166f32.approx_eq(output_sum, (1.0e-2, 2)),
            "Expected sum ~3.166, got {output_sum}"
        );
    }

    /// BatchNorm where scale/bias are static initializers but mean/var are
    /// graph inputs. Should use the Runtime path (no partial lifting).
    #[test]
    fn batch_norm_partial_constant() {
        let model: batch_norm_partial_constant::Model<TestBackend> =
            batch_norm_partial_constant::Model::default();

        let device = Default::default();
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[
                [[0.4967, -0.1383], [0.6477, 1.5230]],
                [[-0.2342, -0.2341], [1.5792, 0.7674]],
                [[-0.4695, 0.5426], [-0.4634, -0.4657]],
            ]],
            &device,
        );
        let mean = Tensor::<TestBackend, 1>::from_floats([0.0, 0.5, -0.5], &device);
        let var = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 0.5], &device);

        let output = model.forward(input, mean, var);

        let expected_shape = Shape::from([1, 3, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        // Same expected sum as batch_norm_runtime (same inputs/params)
        let output_sum = output.sum().into_scalar();
        assert!(
            3.166f32.approx_eq(output_sum, (1.0e-2, 2)),
            "Expected sum ~3.166, got {output_sum}"
        );
    }
}
