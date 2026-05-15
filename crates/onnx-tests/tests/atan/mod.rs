use crate::include_models;
include_models!(atan);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn atan() {
        let device = Default::default();
        let model: atan::Model = atan::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[-2.0, 0.0, 1.0, 5.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[-1.1071f32, 0.0000, 0.7854, 1.3734]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
