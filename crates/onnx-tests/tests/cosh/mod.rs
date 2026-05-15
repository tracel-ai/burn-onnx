// Import the shared macro
use crate::include_models;
include_models!(cosh);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn cosh() {
        let device = Default::default();
        let model: cosh::Model = cosh::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[-4.0, 0.5, 1.0, 9.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[27.3082, 1.1276, 1.5431, 4051.5420]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
