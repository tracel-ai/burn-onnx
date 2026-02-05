use crate::include_models;
include_models!(acos);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn acos() {
        let device = Default::default();
        let model: acos::Model<TestBackend> = acos::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 0.5, -0.5, 1.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[1.5708f32, 1.0472, 2.0944, 0.0000]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
