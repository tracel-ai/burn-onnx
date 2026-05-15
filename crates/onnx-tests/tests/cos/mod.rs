// Import the shared macro
use crate::include_models;
include_models!(cos);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn cos() {
        let device = Default::default();
        let model: cos::Model = cos::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.5403f32, -0.6536, -0.9111, 0.9912]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
