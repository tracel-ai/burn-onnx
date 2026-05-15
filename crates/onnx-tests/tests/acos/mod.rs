use crate::include_models;
include_models!(acos);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn acos() {
        let device = Default::default();
        let model: acos::Model = acos::Model::new(&device);

        let input = Tensor::<4>::from_floats([[[[0.0, 0.5, -0.5, 1.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[1.5708f32, 1.0472, 2.0944, 0.0000]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
