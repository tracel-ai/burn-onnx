use crate::include_models;
include_models!(acosh);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn acosh() {
        let device = Default::default();
        let model: acosh::Model<TestBackend> = acosh::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 2.0, 5.0, 10.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.0000f32, 1.3170, 2.2924, 2.9932]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
