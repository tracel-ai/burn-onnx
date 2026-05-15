// Import the shared macro
use crate::include_models;
include_models!(abs);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn abs() {
        let device = Default::default();
        let model: abs::Model = abs::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[-1.0, -4.0, 9.0, -25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[1.0f32, 4.0, 9.0, 25.0]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
