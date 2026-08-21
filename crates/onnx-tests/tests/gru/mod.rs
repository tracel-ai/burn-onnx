use crate::include_models;
include_models!(
    gru,
    gru_reverse,
    gru_with_initial_state,
    gru_bidirectional,
    gru_runtime_weights,
    gru_bidirectional_runtime_weights,
    gru_bidirectional_static_weights
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Shape, Tensor, TensorData};
    use float_cmp::ApproxEq;

    #[test]
    fn gru_forward() {
        let device = Default::default();
        let model: gru::Model = gru::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4] (seed=99)
        let input = Tensor::<3>::from_floats(
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

        let output_sum = output.sum().into_scalar::<f32>();
        let h_n_sum = h_n.sum().into_scalar::<f32>();

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
        let model: gru_reverse::Model = gru_reverse::Model::default();

        let input = Tensor::<3>::from_floats(
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

        let output_sum = output.sum().into_scalar::<f32>();
        let h_n_sum = h_n.sum().into_scalar::<f32>();

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
        let model: gru_with_initial_state::Model = gru_with_initial_state::Model::default();

        let input = Tensor::<3>::from_floats(
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
        let h_0 = Tensor::<3>::from_floats(
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

        let output_sum = output.sum().into_scalar::<f32>();
        let h_n_sum = h_n.sum().into_scalar::<f32>();

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
        let model: gru_bidirectional::Model = gru_bidirectional::Model::default();

        // Same input as other GRU tests: [seq_length=5, batch_size=2, input_size=4] (seed=99)
        let input = Tensor::<3>::from_floats(
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

        let output_sum = output.sum().into_scalar::<f32>();
        let h_n_sum = h_n.sum().into_scalar::<f32>();

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

    /// The deterministic values `*_runtime_weights.py` feeds the ONNX reference
    /// evaluator: `arange(n) * scale + offset`, reshaped.
    fn ramp<const D: usize>(shape: [usize; D], scale: f32, offset: f32) -> Tensor<D> {
        let count: usize = shape.iter().product();
        let values: alloc::vec::Vec<f32> = (0..count).map(|i| i as f32 * scale + offset).collect();
        Tensor::from_data(TensorData::new(values, shape), &Default::default())
    }

    /// GRU whose `W`/`R`/`B` are graph inputs rather than initializers. Constructed
    /// through `from_file` so the burnpack path is exercised: before this was fixed (#458)
    /// the generated struct declared gate `Param`s that no snapshot filled, and loading
    /// panicked on the missing tensors.
    #[test]
    fn gru_runtime_weights() {
        let device = Default::default();
        let model = gru_runtime_weights::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/gru_runtime_weights.bpk"),
            &device,
        );

        let input = ramp([2, 1, 2], 0.25, -0.5);
        let w = ramp([1, 9, 2], 0.02, -0.4);
        let r = ramp([1, 9, 3], 0.015, -0.3);
        let b = ramp([1, 18], 0.01, -0.1);

        let (y, y_h) = model.forward(input, w, r, b);

        assert_eq!(y.shape(), Shape::from([2, 1, 1, 3]));
        assert_eq!(y_h.shape(), Shape::from([1, 1, 3]));

        // Expected from ONNX ReferenceEvaluator
        let expected_y_sum = 0.130_865_3;
        let expected_y_h_sum = 0.011_934_4;

        let y_sum = y.sum().into_scalar::<f32>();
        let y_h_sum = y_h.sum().into_scalar::<f32>();

        assert!(
            expected_y_sum.approx_eq(y_sum, (1.0e-5, 2)),
            "Y sum mismatch: expected {}, got {}",
            expected_y_sum,
            y_sum
        );
        assert!(
            expected_y_h_sum.approx_eq(y_h_sum, (1.0e-5, 2)),
            "Y_h sum mismatch: expected {}, got {}",
            expected_y_h_sum,
            y_h_sum
        );
    }

    /// Bidirectional GRU with runtime weights, checked against the same weights supplied
    /// as initializers. Nothing upstream covers a bidirectional RNN, so the build-time
    /// snapshot path is the reference and the two must agree element-wise.
    ///
    /// What this pins is the half of the layout that is still written twice: `GateLayout`
    /// drives both paths, so a wrong gate table moves both and stays invisible here, but
    /// `BiasLayout` drives only the emitter while each `collect_*_snapshots` hardcodes its
    /// own bias policy. Flipping GRU to `BiasLayout::Merged` fails this test.
    /// `linear_before_reset=1` is what makes `Wb` and `Rb` separately observable.
    #[test]
    fn gru_bidirectional_runtime_weights_match_initializers() {
        let device = Default::default();
        let runtime = gru_bidirectional_runtime_weights::Model::from_file(
            concat!(
                env!("OUT_DIR"),
                "/model/gru_bidirectional_runtime_weights.bpk"
            ),
            &device,
        );
        let static_weights = gru_bidirectional_static_weights::Model::from_file(
            concat!(
                env!("OUT_DIR"),
                "/model/gru_bidirectional_static_weights.bpk"
            ),
            &device,
        );

        let input = ramp([2, 1, 2], 0.25, -0.5);
        let w = ramp([2, 9, 2], 0.02, -0.4);
        let r = ramp([2, 9, 3], 0.015, -0.3);
        let b = ramp([2, 18], 0.01, -0.1);

        let (y, y_h) = runtime.forward(input.clone(), w, r, b);
        let (expected_y, expected_y_h) = static_weights.forward(input);

        assert_eq!(y.shape(), Shape::from([2, 2, 1, 3]));
        assert_eq!(y_h.shape(), Shape::from([2, 1, 3]));

        y.into_data()
            .assert_approx_eq::<f32>(&expected_y.into_data(), burn::tensor::Tolerance::default());
        y_h.into_data().assert_approx_eq::<f32>(
            &expected_y_h.into_data(),
            burn::tensor::Tolerance::default(),
        );
    }
}
