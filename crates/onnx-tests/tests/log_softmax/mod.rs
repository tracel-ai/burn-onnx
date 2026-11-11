// Import the shared macro
use crate::include_models;
include_models!(
    log_softmax,
    log_softmax_legacy_axis0,
    log_softmax_legacy_default,
    log_softmax_legacy_negative,
    log_softmax_modern_axis0
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn log_softmax() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: log_softmax::Model = log_softmax::Model::new(&device);

        // Run the model
        let input = Tensor::<2>::from_floats(
            [
                [0.33669037, 0.128_809_4, 0.23446237],
                [0.23033303, -1.122_856_4, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [-0.998_838_9f32, -1.206_719_9, -1.101_067],
            [-0.651_105_1, -2.004_294_6, -1.067_766_4],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn legacy_axis0() {
        let device = Device::default();
        let model = log_softmax_legacy_axis0::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/log_softmax_legacy_axis0.bpk"),
            &device,
        );
        let input = Tensor::<2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
        let expected = TensorData::from([[-3.4401897f32, -2.4401897], [-1.4401897, -0.4401897]]);
        model
            .forward(input)
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn legacy_default() {
        let device = Device::default();
        let model = log_softmax_legacy_default::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/log_softmax_legacy_default.bpk"),
            &device,
        );
        let input = Tensor::<3>::from_floats([[[0.0, 1.0], [2.0, 3.0]]], &device);
        let expected = TensorData::from([[[-3.4401897f32, -2.4401897], [-1.4401897, -0.4401897]]]);
        model
            .forward(input)
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn legacy_negative() {
        let device = Device::default();
        let model = log_softmax_legacy_negative::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/log_softmax_legacy_negative.bpk"),
            &device,
        );
        let input = Tensor::<3>::from_floats([[[0.0, 1.0], [2.0, 3.0]]], &device);
        let expected = TensorData::from([[[-3.4401897f32, -2.4401897], [-1.4401897, -0.4401897]]]);
        model
            .forward(input)
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn modern_axis0() {
        let device = Device::default();
        let model = log_softmax_modern_axis0::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/log_softmax_modern_axis0.bpk"),
            &device,
        );
        let input = Tensor::<2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
        let expected = TensorData::from([[-2.126928f32, -2.126928], [-0.12692805, -0.12692805]]);
        model
            .forward(input)
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
