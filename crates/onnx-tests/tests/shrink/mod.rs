// Import the shared macro
use crate::include_models;
include_models!(shrink);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn shrink() {
        // The model contains 1d and 2d shrink nodes
        let model: shrink::Model<TestBackend> = shrink::Model::default();

        let device = Default::default();
        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 5>::from_floats(
            [
                [-2.0, -1.0, 0.0, 1.0, 2.0],
                [-3.0, -2.5, -0.5, 0.5, 3.0],
                [-1.5, 0.0, 1.5, 2.5, 3.5],
                [-2.2, -1.1, 0.0, 1.1, 2.2],
                [-4.0, -2.0, 0.0, 2.0, 4.0],
            ],
            &device,
        );
        let (out_no_bias, out_with_bias) = model.forward(input.clone(), input);

        // From onnx inference
        let expected_no_bias = Tensor::<TestBackend, 5>::from_floats(
            [
                [-2.0, 0.0, 0.0, 0.0, 2.0],
                [-3.0, -2.5, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 2.5, 3.5],
                [-2.2, 0.0, 0.0, 0.0, 2.2],
                [-4.0, -2.0, 0.0, 2.0, 4.0],
            ],
            &device,
        );
        let expected_with_bias = Tensor::<TestBackend, 5>::from_floats(
            [
                [0.5, 0.0, 0.0, 0.0, -0.5],
                [1.5, 1.0, 0.0, 0.0, -1.5],
                [0.0, 0.0, 0.0, -1.0, -2.0],
                [0.70000005, 0.0, 0.0, 0.0, -0.70000005],
                [2.5, 1.5, 1.5, -1.5, -2.5],
            ],
            &device,
        );

        assert_eq!(out_no_bias, expected_no_bias);
        assert_eq!(out_with_bias, expected_with_bias);
    }
}
