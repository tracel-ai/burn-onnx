use crate::include_models;
include_models!(softsign);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn softsign() {
        let device = Default::default();
        let model: softsign::Model<TestBackend> = softsign::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857],
                [1.5230298, -0.23415338, -0.23413695],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.331_869_75f32, -0.121_469_42, 0.393_089_18],
            [0.603_651_1, -0.189_727_93, -0.189_717_16],
        ]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
