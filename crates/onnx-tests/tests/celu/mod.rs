use crate::include_models;
include_models!(celu);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn celu() {
        let device = Default::default();
        let model: celu::Model<TestBackend> = celu::Model::new(&device);

        // Run the model (alpha=2.0)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857],
                [1.5230298, -0.23415338, -0.23413695],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.496_714_14f32, -0.133_593_32, 0.647_688_57],
            [1.523_029_8, -0.220_966_1, -0.220_951_44],
        ]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
