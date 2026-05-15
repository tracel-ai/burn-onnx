use crate::include_models;
include_models!(
    mel_weight_matrix_constants,
    mel_weight_matrix_runtime,
    mel_weight_matrix_shaped
);

#[cfg(test)]
mod tests {
    use super::*;

    // Expected for num_mel_bins=8, dft_length=16, sample_rate=8192, lower=0, upper=4096.
    // This is the exact configuration used by the official ONNX test_melweightmatrix case;
    // the triangles collapse to peak-only columns because the mel edges align with DFT bins.
    const OFFICIAL_EXPECTED: [[f32; 8]; 9] = [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    #[test]
    fn mel_weight_matrix_constants() {
        let device = Default::default();
        let model: mel_weight_matrix_constants::Model =
            mel_weight_matrix_constants::Model::new(&device);

        let output = model.forward();
        let expected = burn::tensor::Tensor::<2>::from_floats(OFFICIAL_EXPECTED, &device);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn mel_weight_matrix_runtime() {
        let device = Default::default();
        let model: mel_weight_matrix_runtime::Model =
            mel_weight_matrix_runtime::Model::new(&device);

        let output = model.forward(8, 16, 8192, 0.0, 4096.0);
        let expected = burn::tensor::Tensor::<2>::from_floats(OFFICIAL_EXPECTED, &device);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn mel_weight_matrix_shaped() {
        let device = Default::default();
        let model: mel_weight_matrix_shaped::Model = mel_weight_matrix_shaped::Model::new(&device);

        // num_mel_bins=4, dft_length=32, sample_rate=16000, 300-4000 Hz.
        let output = model.forward(4, 32, 16000, 300.0, 4000.0);

        // Expected [17, 4] from ONNX reference evaluator.
        let expected = burn::tensor::Tensor::<2>::from_floats(
            [
                [0.0_f32, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            &device,
        );

        output.to_data().assert_approx_eq::<f32>(
            &expected.to_data(),
            burn::tensor::Tolerance::rel_abs(1e-3, 1e-4),
        );
    }
}
