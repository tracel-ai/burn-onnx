use crate::include_models;
include_models!(hann_window, hann_window_symmetric, hann_window_runtime);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::TestBackend;

    #[test]
    fn hann_window_periodic() {
        let device = Default::default();
        let model: hann_window::Model<TestBackend> = hann_window::Model::new(&device);

        let output = model.forward();

        // Expected periodic Hann window of size 10 (from ONNX reference evaluator)
        let expected = burn::tensor::Tensor::<TestBackend, 1>::from_floats(
            [
                0.0_f32,
                0.095_491_506,
                0.345_491_53,
                0.654_508_5,
                0.904_508_5,
                1.0,
                0.904_508_5,
                0.654_508_4,
                0.345_491_44,
                0.095_491_5,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn hann_window_symmetric_test() {
        let device = Default::default();
        let model: hann_window_symmetric::Model<TestBackend> =
            hann_window_symmetric::Model::new(&device);

        let output = model.forward();

        // Expected symmetric Hann window of size 10 (from ONNX reference evaluator)
        let expected = burn::tensor::Tensor::<TestBackend, 1>::from_floats(
            [
                0.0_f32,
                0.116_977_78,
                0.413_175_94,
                0.75,
                0.969_846_3,
                0.969_846_3,
                0.75,
                0.413_175_76,
                0.116_977_73,
                0.0,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn hann_window_runtime_size() {
        let device = Default::default();
        let model: hann_window_runtime::Model<TestBackend> =
            hann_window_runtime::Model::new(&device);

        // Pass size=10 at runtime
        let output = model.forward(10);

        let expected = burn::tensor::Tensor::<TestBackend, 1>::from_floats(
            [
                0.0_f32,
                0.095_491_506,
                0.345_491_53,
                0.654_508_5,
                0.904_508_5,
                1.0,
                0.904_508_5,
                0.654_508_4,
                0.345_491_44,
                0.095_491_5,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
