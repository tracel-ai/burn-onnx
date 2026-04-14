use crate::include_models;
include_models!(
    imputer,
    imputer_per_feature,
    imputer_int,
    imputer_nan,
    imputer_nan_default
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData, Tolerance};

    #[test]
    fn imputer_sentinel_replacement() {
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
    fn imputer_per_feature_sentinel_replacement() {
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

    #[test]
    fn imputer_int_sentinel_replacement() {
        let device = Default::default();
        let model: imputer_int::Model<TestBackend> = imputer_int::Model::new(&device);

        // input: [[1, -1, 3], [4, 5, -1]]
        // -1 is the sentinel replaced by 0 (imputed value)
        // Note: Using i32 literals to match backend's default Int type
        let input = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Int>::from_ints(
            [[1i32, -1, 3], [4, 5, -1]],
            &device,
        );

        let output = model.forward(input);

        let expected = burn::tensor::TensorData::from([[1i32, 0, 3], [4, 5, 0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn imputer_nan_sentinel_replacement() {
        let device = Default::default();
        let model: imputer_nan::Model<TestBackend> = imputer_nan::Model::new(&device);

        // input: [[1.0, NaN, 3.0], [4.0, 5.0, NaN]]
        // NaN is the explicit sentinel replaced by 0.0 (imputed value)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [[1.0f32, f32::NAN, 3.0], [4.0, 5.0, f32::NAN]],
            &device,
        );

        let output = model.forward(input);

        let expected = TensorData::from([[1.0f32, 0.0, 3.0], [4.0, 5.0, 0.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn imputer_nan_default_sentinel_replacement() {
        let device = Default::default();
        let model: imputer_nan_default::Model<TestBackend> =
            imputer_nan_default::Model::new(&device);

        // input: [[1.0, NaN, 3.0], [4.0, 5.0, NaN]]
        // NaN is the DEFAULT sentinel (replaced_value_float attribute omitted)
        // replaced by 0.0 (imputed value)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [[1.0f32, f32::NAN, 3.0], [4.0, 5.0, f32::NAN]],
            &device,
        );

        let output = model.forward(input);

        let expected = TensorData::from([[1.0f32, 0.0, 3.0], [4.0, 5.0, 0.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
