use crate::include_models;
include_models!(
    blackman_window,
    blackman_window_symmetric,
    blackman_window_runtime
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackman_window_periodic() {
        let device = Default::default();
        let model: blackman_window::Model = blackman_window::Model::new(&device);

        let output = model.forward();

        // Expected periodic Blackman window of size 10 (from ONNX reference evaluator)
        let expected = burn::tensor::Tensor::<1>::from_floats(
            [
                0.0_f32,
                0.040_212_862,
                0.200_770_14,
                0.509_787_14,
                0.849_229_87,
                1.0,
                0.849_229_87,
                0.509_787_14,
                0.200_770_14,
                0.040_212_862,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn blackman_window_symmetric_test() {
        let device = Default::default();
        let model: blackman_window_symmetric::Model =
            blackman_window_symmetric::Model::new(&device);

        let output = model.forward();

        // Expected symmetric Blackman window of size 10 (from ONNX reference evaluator)
        let expected = burn::tensor::Tensor::<1>::from_floats(
            [
                0.0_f32,
                0.050_869_63,
                0.258_000_49,
                0.63,
                0.951_129_85,
                0.951_129_85,
                0.63,
                0.258_000_49,
                0.050_869_63,
                0.0,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn blackman_window_runtime_size() {
        let device = Default::default();
        let model: blackman_window_runtime::Model = blackman_window_runtime::Model::new(&device);

        // Pass size=10 at runtime
        let output = model.forward(10);

        // Expected periodic Blackman window of size 10 (from ONNX reference evaluator)
        let expected = burn::tensor::Tensor::<1>::from_floats(
            [
                0.0_f32,
                0.040_212_862,
                0.200_770_14,
                0.509_787_14,
                0.849_229_87,
                1.0,
                0.849_229_87,
                0.509_787_14,
                0.200_770_14,
                0.040_212_862,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
