use crate::include_models;
include_models!(
    hamming_window,
    hamming_window_symmetric,
    hamming_window_runtime
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::TestBackend;

    #[test]
    fn hamming_window_periodic() {
        let device = Default::default();
        let model: hamming_window::Model<TestBackend> = hamming_window::Model::new(&device);

        let output = model.forward();

        // Expected periodic Hamming window of size 10 (from ONNX reference evaluator)
        let expected = burn::tensor::Tensor::<TestBackend, 1>::from_floats(
            [
                0.086_956_52_f32,
                0.174_144_42,
                0.402_405_3,
                0.684_551_24,
                0.912_812_1,
                1.0,
                0.912_812_1,
                0.684_551_1,
                0.402_405_23,
                0.174_144_42,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn hamming_window_symmetric_test() {
        let device = Default::default();
        let model: hamming_window_symmetric::Model<TestBackend> =
            hamming_window_symmetric::Model::new(&device);

        let output = model.forward();

        // Expected symmetric Hamming window of size 10 (from ONNX reference evaluator)
        let expected = burn::tensor::Tensor::<TestBackend, 1>::from_floats(
            [
                0.086_956_52_f32,
                0.193_762_33,
                0.464_204_1,
                0.771_739_13,
                0.972_468_4,
                0.972_468_4,
                0.771_739_13,
                0.464_203_95,
                0.193_762_27,
                0.086_956_52,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn hamming_window_runtime_size() {
        let device = Default::default();
        let model: hamming_window_runtime::Model<TestBackend> =
            hamming_window_runtime::Model::new(&device);

        // Pass size=10 at runtime
        let output = model.forward(10);

        let expected = burn::tensor::Tensor::<TestBackend, 1>::from_floats(
            [
                0.086_956_52_f32,
                0.174_144_42,
                0.402_405_3,
                0.684_551_24,
                0.912_812_1,
                1.0,
                0.912_812_1,
                0.684_551_1,
                0.402_405_23,
                0.174_144_42,
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
