use crate::include_models;
include_models!(atan);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn atan() {
        let device = Default::default();
        let model: atan::Model<TestBackend> = atan::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats([[[[-2.0, 0.0, 1.0, 5.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[-1.1071f32, 0.0000, 0.7854, 1.3734]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
