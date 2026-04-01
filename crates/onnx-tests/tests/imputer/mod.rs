use crate::include_models;
include_models!(imputer, imputer_per_feature);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData, Tolerance};

    #[test]
    fn imputer_nan_replacement() {
        let device = Default::default();
        let model: imputer::Model<TestBackend> = imputer::Model::new(&device);

        // input: [[1.0, -999.0, 3.0], [4.0, 5.0, -999.0]]
        // -999.0 is replaced by 0.0 (imputed value)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [[1.0f32, -999.0, 3.0], [4.0, 5.0, -999.0]],
            &device,
        );

        let output = model.forward(input);

        let expected = TensorData::from([[1.0f32, 0.0, 3.0], [4.0, 5.0, 0.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn imputer_per_feature_nan_replacement() {
        let device = Default::default();
        let model: imputer_per_feature::Model<TestBackend> =
            imputer_per_feature::Model::new(&device);

        // input: [[-999.0, 2.0, -999.0], [4.0, -999.0, 6.0]]
        // per-feature imputed values: [10.0, 20.0, 30.0]
        let input = Tensor::<TestBackend, 2>::from_floats(
            [[-999.0f32, 2.0, -999.0], [4.0, -999.0, 6.0]],
            &device,
        );

        let output = model.forward(input);

        let expected = TensorData::from([[10.0f32, 2.0, 30.0], [4.0, 20.0, 6.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
