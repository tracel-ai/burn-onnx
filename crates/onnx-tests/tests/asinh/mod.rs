use crate::include_models;
include_models!(asinh);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn asinh() {
        let device = Default::default();
        let model: asinh::Model<TestBackend> = asinh::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats([[[[-4.0, 0.0, 1.0, 9.0]]]], &device);

        let output = model.forward(input);
        let expected = TensorData::from([[[[-2.0947f32, 0.0000, 0.8814, 2.8934]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
