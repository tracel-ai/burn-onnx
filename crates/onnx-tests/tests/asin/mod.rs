use crate::include_models;
include_models!(asin);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn asin() {
        let device = Default::default();
        let model: asin::Model = asin::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[0.0, 0.5, -0.5, 1.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.0000f32, 0.5236, -0.5236, 1.5708]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
