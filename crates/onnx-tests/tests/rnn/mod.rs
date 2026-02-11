use crate::include_models;
include_models!(rnn, rnn_bidirectional, rnn_reverse, rnn_with_initial_state);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn rnn_forward() {
        let device = Default::default();
        // Initialize the model with weights (loaded from the exported file)
        let model: rnn::Model<TestBackend> = rnn::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4]
        // Using random values matching the PyTorch test
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-1.4570, -0.1023, -0.5992, 0.4771],
                    [0.7262, 0.0912, -0.3891, 0.5279],
                ],
                [
                    [-0.0127, 0.2408, 0.1325, 0.7642],
                    [1.0950, 0.3399, 0.7200, 0.4114],
                ],
                [
                    [1.9312, 1.0119, -1.4364, -1.1299],
                    [-0.1360, 1.6354, 0.6547, 0.5760],
                ],
                [
                    [1.0414, -0.3997, -2.2933, 0.4976],
                    [-0.4257, -1.3371, -0.1933, 0.6526],
                ],
                [
                    [-0.3063, -0.3302, -0.9808, 0.1947],
                    [-1.6535, 0.6814, 1.4611, -0.3098],
                ],
            ],
            &device,
        );

        let (output, h_n) = model.forward(input);

        // Test output shapes
        // ONNX model has a Squeeze node that removes the num_directions dimension
        // Output: [seq_length, batch_size, hidden_size] = [5, 2, 8]
        let expected_output_shape = Shape::from([5, 2, 8]);
        // h_n: [num_directions, batch_size, hidden_size] = [1, 2, 8]
        let expected_h_n_shape = Shape::from([1, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_n_shape);

        // Verify approximate output values using sum (values from PyTorch)
        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();

        // Expected values from ONNX runtime inference
        let expected_output_sum = -5.624_653_8;
        let expected_h_n_sum = -1.624_171_9;

        assert!(
            expected_output_sum.approx_eq(output_sum, (1.0e-4, 2)),
            "Output sum mismatch: expected {}, got {}",
            expected_output_sum,
            output_sum
        );
        assert!(
            expected_h_n_sum.approx_eq(h_n_sum, (1.0e-4, 2)),
            "h_n sum mismatch: expected {}, got {}",
            expected_h_n_sum,
            h_n_sum
        );
    }

    #[test]
    fn rnn_bidirectional_forward() {
        let device = Default::default();
        let model: rnn_bidirectional::Model<TestBackend> = rnn_bidirectional::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4]
        // Using random values matching the PyTorch test (seed 42)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-4.5051e-03, 1.6668e+00, 1.5392e-01, -1.0603e+00],
                    [-5.7266e-01, 8.3568e-02, 3.9991e-01, 1.9892e+00],
                ],
                [
                    [-7.1988e-02, -9.0609e-01, -2.0487e+00, -1.0811e+00],
                    [1.7623e-02, 7.8226e-02, 1.9316e-01, 4.0967e-01],
                ],
                [
                    [-9.2913e-01, 2.7619e-01, -5.3888e-01, 4.6258e-01],
                    [-8.7189e-01, -2.7118e-02, -3.5325e-01, 1.4639e+00],
                ],
                [
                    [5.6351e-01, 1.8582e+00, 1.0441e+00, -8.6382e-01],
                    [8.3509e-01, -3.1571e-01, 2.6911e-01, 8.5404e-02],
                ],
                [
                    [-1.4129e+00, -1.8791e+00, -1.7983e-01, 7.9039e-01],
                    [5.2394e-01, -2.6935e-01, -1.6191e+00, 1.2588e-03],
                ],
            ],
            &device,
        );

        let (output, h_n) = model.forward(input);

        // Test output shapes for bidirectional rnn
        // Output: [seq_length, batch_size, 2*hidden_size] = [5, 2, 16]
        let expected_output_shape = Shape::from([5, 2, 16]);
        // h_n: [num_directions, batch_size, hidden_size] = [2, 2, 8]
        let expected_h_shape = Shape::from([2, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);

        // Verify approximate output values using sum (values from PyTorch)
        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();

        // Expected values from PyTorch inference
        let expected_output_sum = -10.244_780_5;
        let expected_h_n_sum = -4.249_545_5;

        assert!(
            expected_output_sum.approx_eq(output_sum, (1.0e-4, 2)),
            "Output sum mismatch: expected {}, got {}",
            expected_output_sum,
            output_sum
        );
        assert!(
            expected_h_n_sum.approx_eq(h_n_sum, (1.0e-4, 2)),
            "h_n sum mismatch: expected {}, got {}",
            expected_h_n_sum,
            h_n_sum
        );
    }

    #[test]
    fn rnn_reverse_forward() {
        let device = Default::default();
        let model: rnn_reverse::Model<TestBackend> = rnn_reverse::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4]
        // Using random values matching the PyTorch test (seed 42)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-1.4570, -0.1023, -0.5992, 0.4771],
                    [0.7262, 0.0912, -0.3891, 0.5279],
                ],
                [
                    [-0.0127, 0.2408, 0.1325, 0.7642],
                    [1.0950, 0.3399, 0.7200, 0.4114],
                ],
                [
                    [1.9312, 1.0119, -1.4364, -1.1299],
                    [-0.1360, 1.6354, 0.6547, 0.5760],
                ],
                [
                    [1.0414, -0.3997, -2.2933, 0.4976],
                    [-0.4257, -1.3371, -0.1933, 0.6526],
                ],
                [
                    [-0.3063, -0.3302, -0.9808, 0.1947],
                    [-1.6535, 0.6814, 1.4611, -0.3098],
                ],
            ],
            &device,
        );

        let (output, h_n) = model.forward(input);

        // Test output shapes for reverse rnn
        // Output: [seq_length, batch_size, hidden_size] = [5, 2, 8]
        // ONNX model has a Squeeze node that removes the num_directions dimension
        let expected_output_shape = Shape::from([5, 2, 8]);
        // h_n: [num_directions, batch_size, hidden_size] = [1, 2, 8]
        let expected_h_shape = Shape::from([1, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);

        // Verify approximate output values using sum (values from PyTorch)
        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();

        // Expected values from PyTorch inference with reverse direction simulation
        let expected_output_sum = -5.572_328_5;
        let expected_h_n_sum = -1.168_946_5;

        assert!(
            expected_output_sum.approx_eq(output_sum, (1.0e-4, 2)),
            "Output sum mismatch: expected {}, got {}",
            expected_output_sum,
            output_sum
        );
        assert!(
            expected_h_n_sum.approx_eq(h_n_sum, (1.0e-4, 2)),
            "h_n sum mismatch: expected {}, got {}",
            expected_h_n_sum,
            h_n_sum
        );
    }

    #[test]
    fn rnn_with_initial_state_forward() {
        let device = Default::default();
        let model: rnn_with_initial_state::Model<TestBackend> =
            rnn_with_initial_state::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4]
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-1.4570, -0.1023, -0.5992, 0.4771],
                    [0.7262, 0.0912, -0.3891, 0.5279],
                ],
                [
                    [-0.0127, 0.2408, 0.1325, 0.7642],
                    [1.0950, 0.3399, 0.7200, 0.4114],
                ],
                [
                    [1.9312, 1.0119, -1.4364, -1.1299],
                    [-0.1360, 1.6354, 0.6547, 0.5760],
                ],
                [
                    [1.0414, -0.3997, -2.2933, 0.4976],
                    [-0.4257, -1.3371, -0.1933, 0.6526],
                ],
                [
                    [-0.3063, -0.3302, -0.9808, 0.1947],
                    [-1.6535, 0.6814, 1.4611, -0.3098],
                ],
            ],
            &device,
        );

        // Initial hidden state: [num_directions=1, batch_size=2, hidden_size=8]
        let h_0 = Tensor::<TestBackend, 3>::from_floats(
            [[
                [
                    0.9633, -0.3095, 0.5712, 1.1179, -1.2956, 0.0503, -0.5855, -0.3900,
                ],
                [
                    0.9812, -0.6401, -0.4908, 0.2080, -1.1586, -0.9637, -0.3750, 0.8033,
                ],
            ]],
            &device,
        );

        let (output, h_n) = model.forward(input, h_0);

        // Test output shapes
        // ONNX model has a Squeeze node that removes the num_directions dimension
        let expected_output_shape = Shape::from([5, 2, 8]);
        let expected_h_shape = Shape::from([1, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);

        // Verify approximate output values using sum (values from PyTorch)
        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();

        // Expected values from PyTorch inference with initial states
        let expected_output_sum = -4.966_948_9;
        let expected_h_n_sum = -1.581_394_9;

        // Use slightly looser tolerance for initial state test due to accumulated floating point differences
        assert!(
            expected_output_sum.approx_eq(output_sum, (1.0e-3, 2)),
            "Output sum mismatch: expected {}, got {}",
            expected_output_sum,
            output_sum
        );
        assert!(
            expected_h_n_sum.approx_eq(h_n_sum, (1.0e-3, 2)),
            "h_n sum mismatch: expected {}, got {}",
            expected_h_n_sum,
            h_n_sum
        );
    }
}
