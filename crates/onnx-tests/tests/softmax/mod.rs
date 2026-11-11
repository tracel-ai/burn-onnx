// Import the shared macro
use crate::include_models;
include_models!(
    softmax,
    softmax_legacy_axis0,
    softmax_legacy_default,
    softmax_legacy_negative,
    softmax_modern_axis0
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn softmax() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: softmax::Model = softmax::Model::new(&device);

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
            [0.36830685f32, 0.29917702, 0.33251613],
            [0.521_469_2, 0.13475533, 0.343_775_5],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn legacy_axis0() {
        let device = Device::default();
        let model = softmax_legacy_axis0::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/softmax_legacy_axis0.bpk"),
            &device,
        );
        let input = Tensor::<2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
        let expected = TensorData::from([[0.032058604f32, 0.08714432], [0.23688282, 0.6439143]]);
        model
            .forward(input)
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn legacy_default() {
        let device = Device::default();
        let model = softmax_legacy_default::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/softmax_legacy_default.bpk"),
            &device,
        );
        let input = Tensor::<3>::from_floats([[[0.0, 1.0], [2.0, 3.0]]], &device);
        let expected = TensorData::from([[[0.032058604f32, 0.08714432], [0.23688282, 0.6439143]]]);
        model
            .forward(input)
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn legacy_negative() {
        let device = Device::default();
        let model = softmax_legacy_negative::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/softmax_legacy_negative.bpk"),
            &device,
        );
        let input = Tensor::<3>::from_floats([[[0.0, 1.0], [2.0, 3.0]]], &device);
        let expected = TensorData::from([[[0.032058604f32, 0.08714432], [0.23688282, 0.6439143]]]);
        model
            .forward(input)
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn modern_axis0() {
        let device = Device::default();
        let model = softmax_modern_axis0::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/softmax_modern_axis0.bpk"),
            &device,
        );
        let input = Tensor::<2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
        let expected = TensorData::from([[0.11920292f32, 0.11920292], [0.8807971, 0.8807971]]);
        model
            .forward(input)
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
