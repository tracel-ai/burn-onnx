use crate::include_models;
include_models!(selu, selu_custom);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn selu() {
        let device = Default::default();
        let model: selu::Model = selu::Model::new(&device);

        let input = Tensor::<2>::from_floats([[-1.0, 0.0, 1.0], [2.0, -0.5, -2.0]], &device);
        let output = model.forward(input);
        let expected = TensorData::from([
            [-1.111_330_6f32, 0.0, 1.050_701],
            [2.101_402, -0.691_758_2, -1.520_166_5],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn selu_custom_alpha_gamma() {
        let device = Default::default();
        let model: selu_custom::Model = selu_custom::Model::new(&device);

        let input = Tensor::<2>::from_floats([[-1.0, 0.0, 1.0], [2.0, -0.5, -2.0]], &device);
        let output = model.forward(input);
        let expected =
            TensorData::from([[-3.792_723_2f32, 0.0, 3.0], [6.0, -2.360_816, -5.187_988_3]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
