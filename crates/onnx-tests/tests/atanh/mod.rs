use crate::include_models;
include_models!(atanh);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn atanh() {
        let device = Default::default();
        let model: atanh::Model = atanh::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[-0.5, 0.0, 0.5, 0.9]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[-0.5493f32, 0.0000, 0.5493, 1.4722]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
