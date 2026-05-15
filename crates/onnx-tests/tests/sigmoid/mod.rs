use crate::include_models;
include_models!(sigmoid);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn sigmoid() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: sigmoid::Model = sigmoid::Model::new(&device);

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
            [0.58338636f32, 0.532_157_9, 0.55834854],
            [0.557_33, 0.24548186, 0.45355222],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
