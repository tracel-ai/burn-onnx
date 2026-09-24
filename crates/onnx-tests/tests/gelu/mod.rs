// Import the shared macro
use crate::include_models;
include_models!(gelu, gelu_tanh);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn gelu() {
        let device = Default::default();
        let model: gelu::Model = gelu::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.8413447f32, 3.9998732, 9.0, 25.0]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn gelu_tanh() {
        let device = Default::default();
        let model: gelu_tanh::Model = gelu_tanh::Model::new(&device);

        let input = Tensor::<1>::from_floats([-3.0, -1.5, 0.5, 2.0], &device);

        let output = model.forward(input);
        // The exact erf form differs from these by up to 4e-4.
        let expected = TensorData::from([-0.0036373436f32, -0.10042843, 0.345714, 1.9545977]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-5));
    }
}
