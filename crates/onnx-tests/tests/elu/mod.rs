use crate::include_models;
include_models!(elu);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn elu() {
        let device = Default::default();
        let model: elu::Model = elu::Model::new(&device);

        // Run the model (alpha=0.5)
        let input = Tensor::<2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857],
                [1.5230298, -0.23415338, -0.23413695],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.496_714_14f32, -0.064_565_75, 0.647_688_57],
            [1.523_029_8, -0.104_379_77, -0.104_373_28],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
