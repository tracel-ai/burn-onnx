// Import the shared macro
use crate::include_models;
include_models!(sign);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn sign() {
        let device = Default::default();
        let model: sign::Model = sign::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[-1.0, 2.0, 0.0, -4.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[-1.0f32, 1.0, 0.0, -1.0]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
