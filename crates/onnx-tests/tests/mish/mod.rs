// Import the shared macro
use crate::include_models;
include_models!(mish);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn mish() {
        let device = Default::default();
        let model: mish::Model = mish::Model::new(&device);

        let input = Tensor::<2>::from_floats([[1.0, -1.0, 0.0, 5.0]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[0.8650984f32, -0.30340144, 0.0, 4.9995522]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
