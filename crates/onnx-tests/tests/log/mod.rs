use crate::include_models;
include_models!(log);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn log() {
        let device = Default::default();
        let model: log::Model = log::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.0000f32, 1.3863, 2.1972, 3.2189]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
