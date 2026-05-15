use crate::include_models;
include_models!(tanh);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn tanh() {
        // Initialize the model
        let device = Default::default();
        let model = tanh::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let output = model.forward(input);
        // data from pyTorch
        let expected = TensorData::from([[[[0.7616f32, 0.9640, 0.9951, 0.9993]]]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
