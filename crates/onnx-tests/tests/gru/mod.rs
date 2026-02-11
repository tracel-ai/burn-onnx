use crate::include_models;
include_models!(gru, gru_reverse, gru_with_initial_state, gru_bidirectional);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn gru_forward() {
        let device = Default::default();
        let model: gru::Model<TestBackend> = gru::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4] (seed=99)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-0.1424, 2.0572, 0.2833, 1.3298],
                    [-0.1546, -0.0690, 0.7552, 0.8256],
                ],
                [
                    [-0.1131, -2.3678, -0.1670, 0.6854],
                    [0.0235, 0.4562, 0.2705, -1.4350],
                ],
                [
                    [0.8828, -0.5801, -0.5016, 0.5910],
                    [-0.7316, 0.2618, -0.8558, -0.1875],
                ],
                [
                    [-0.3735, -0.4620, -0.8165, -0.0451],
                    [0.1213, 0.9260, -0.5738, 0.0527],
                ],
                [
                    [2.2073, 0.3918, 0.4827, 0.4333],
                    [-1.7043, -0.2439, -2.1397, 0.8613],
                ],
            ],
            &device,
        );

        let (output, h_n) = model.forward(input);

        // Y: [seq_length, num_directions, batch_size, hidden_size] = [5, 1, 2, 8]
        let expected_output_shape = Shape::from([5, 1, 2, 8]);
        // Y_h: [num_directions, batch_size, hidden_size] = [1, 2, 8]
        let expected_h_shape = Shape::from([1, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);

        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();

        // Expected from ONNX ReferenceEvaluator
        let expected_output_sum = -0.050_507_9;
        let expected_h_n_sum = 0.282_824_7;

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
    fn gru_reverse_forward() {
        let device = Default::default();
        let model: gru_reverse::Model<TestBackend> = gru_reverse::Model::default();

        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-0.1424, 2.0572, 0.2833, 1.3298],
                    [-0.1546, -0.0690, 0.7552, 0.8256],
                ],
                [
                    [-0.1131, -2.3678, -0.1670, 0.6854],
                    [0.0235, 0.4562, 0.2705, -1.4350],
                ],
                [
                    [0.8828, -0.5801, -0.5016, 0.5910],
                    [-0.7316, 0.2618, -0.8558, -0.1875],
                ],
                [
                    [-0.3735, -0.4620, -0.8165, -0.0451],
                    [0.1213, 0.9260, -0.5738, 0.0527],
                ],
                [
                    [2.2073, 0.3918, 0.4827, 0.4333],
                    [-1.7043, -0.2439, -2.1397, 0.8613],
                ],
            ],
            &device,
        );

        let (output, h_n) = model.forward(input);

        // Y: [seq_length, num_directions, batch_size, hidden_size] = [5, 1, 2, 8]
        let expected_output_shape = Shape::from([5, 1, 2, 8]);
        // Y_h: [num_directions, batch_size, hidden_size] = [1, 2, 8]
        let expected_h_shape = Shape::from([1, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);

        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();

        let expected_output_sum = -2.035_414_7;
        let expected_h_n_sum = -0.936_855_7;

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
    fn gru_with_initial_state_forward() {
        let device = Default::default();
        let model: gru_with_initial_state::Model<TestBackend> =
            gru_with_initial_state::Model::default();

        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-0.1424, 2.0572, 0.2833, 1.3298],
                    [-0.1546, -0.0690, 0.7552, 0.8256],
                ],
                [
                    [-0.1131, -2.3678, -0.1670, 0.6854],
                    [0.0235, 0.4562, 0.2705, -1.4350],
                ],
                [
                    [0.8828, -0.5801, -0.5016, 0.5910],
                    [-0.7316, 0.2618, -0.8558, -0.1875],
                ],
                [
                    [-0.3735, -0.4620, -0.8165, -0.0451],
                    [0.1213, 0.9260, -0.5738, 0.0527],
                ],
                [
                    [2.2073, 0.3918, 0.4827, 0.4333],
                    [-1.7043, -0.2439, -2.1397, 0.8613],
                ],
            ],
            &device,
        );

        // Initial hidden state: [num_directions=1, batch_size=2, hidden_size=8]
        let h_0 = Tensor::<TestBackend, 3>::from_floats(
            [[
                [
                    0.8501, -0.2644, 0.8817, -0.5608, -0.5960, 0.2764, -0.4080, -0.2483,
                ],
                [
                    0.5431, -0.4873, -0.1411, -0.0586, 0.1893, 0.3661, -0.0518, -0.5994,
                ],
            ]],
            &device,
        );

        let (output, h_n) = model.forward(input, h_0);

        let expected_output_shape = Shape::from([5, 1, 2, 8]);
        let expected_h_shape = Shape::from([1, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);

        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();

        let expected_output_sum = 5.103_189_0;
        let expected_h_n_sum = 0.052_935_8;

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

    #[test]
    fn gru_bidirectional_forward() {
        let device = Default::default();
        let model: gru_bidirectional::Model<TestBackend> = gru_bidirectional::Model::default();

        // Same input as other GRU tests: [seq_length=5, batch_size=2, input_size=4] (seed=99)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-0.1424, 2.0572, 0.2833, 1.3298],
                    [-0.1546, -0.0690, 0.7552, 0.8256],
                ],
                [
                    [-0.1131, -2.3678, -0.1670, 0.6854],
                    [0.0235, 0.4562, 0.2705, -1.4350],
                ],
                [
                    [0.8828, -0.5801, -0.5016, 0.5910],
                    [-0.7316, 0.2618, -0.8558, -0.1875],
                ],
                [
                    [-0.3735, -0.4620, -0.8165, -0.0451],
                    [0.1213, 0.9260, -0.5738, 0.0527],
                ],
                [
                    [2.2073, 0.3918, 0.4827, 0.4333],
                    [-1.7043, -0.2439, -2.1397, 0.8613],
                ],
            ],
            &device,
        );

        let (output, h_n) = model.forward(input);

        // Y: [seq_length, num_directions, batch_size, hidden_size] = [5, 2, 2, 8]
        let expected_output_shape = Shape::from([5, 2, 2, 8]);
        // Y_h: [num_directions, batch_size, hidden_size] = [2, 2, 8]
        let expected_h_shape = Shape::from([2, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);

        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();

        // Expected from onnxruntime
        let expected_output_sum = 1.811_869_7;
        let expected_h_n_sum = 0.934_044_6;

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
