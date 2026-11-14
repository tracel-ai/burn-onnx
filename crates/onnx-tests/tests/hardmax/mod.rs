use crate::include_models;
include_models!(hardmax, hardmax_legacy);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn legacy_axis_zero_normalizes_the_whole_tensor() {
        let device = Default::default();
        let model: hardmax_legacy::Model = hardmax_legacy::Model::new(&device);
        let input = Tensor::<2>::from_floats([[0., 1., 2.], [3., 4., 5.]], &device);
        let output = model.forward(input);
        output.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[0.0f32, 0., 0.], [0., 0., 1.]]),
            burn::tensor::Tolerance::default(),
        );
    }

    #[test]
    fn hardmax() {
        let device = Default::default();
        let model: hardmax::Model = hardmax::Model::new(&device);

        let input = Tensor::<2>::from_floats(
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
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
