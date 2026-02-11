use crate::include_models;
include_models!(selu);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn selu() {
        let device = Default::default();
        let model: selu::Model<TestBackend> = selu::Model::new(&device);

        let input =
            Tensor::<TestBackend, 2>::from_floats([[-1.0, 0.0, 1.0], [2.0, -0.5, -2.0]], &device);
        let output = model.forward(input);
        let expected = TensorData::from([
            [-1.111_330_6f32, 0.0, 1.050_701],
            [2.101_402, -0.691_758_2, -1.520_166_5],
        ]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
