use crate::include_models;
include_models!(recip);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn recip() {
        // Initialize the model
        let device = Default::default();
        let model = recip::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let output = model.forward(input);
        // data from pyTorch
        let expected = TensorData::from([[[[1.0000f32, 0.5000, 0.3333, 0.2500]]]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
