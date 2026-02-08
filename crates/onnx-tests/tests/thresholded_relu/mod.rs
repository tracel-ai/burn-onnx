use crate::include_models;
include_models!(thresholded_relu);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn thresholded_relu() {
        let device = Default::default();
        let model: thresholded_relu::Model<TestBackend> = thresholded_relu::Model::new(&device);

        // Run the model (alpha=2.0)
        // Includes boundary case: 2.0 == alpha should output 0.0 (strict > comparison)
        let input =
            Tensor::<TestBackend, 2>::from_floats([[-1.0, 0.0, 2.0], [2.5, 1.5, 5.0]], &device);
        let output = model.forward(input);
        let expected = TensorData::from([[0.0f32, 0.0, 0.0], [2.5, 0.0, 5.0]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
