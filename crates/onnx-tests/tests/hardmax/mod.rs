use crate::include_models;
include_models!(hardmax);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn hardmax() {
        let device = Default::default();
        let model: hardmax::Model<TestBackend> = hardmax::Model::new(&device);

        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857],
                [1.5230298, -0.23415338, -0.23413695],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[0.0f32, 0.0, 1.0], [1.0, 0.0, 0.0]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
