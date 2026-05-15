// Import the shared macro
use crate::include_models;
include_models!(exp);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    #[allow(clippy::approx_constant)]
    fn exp() {
        let device = Default::default();
        let model: exp::Model = exp::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[0.0000, 0.6931]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[1f32, 2.]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
