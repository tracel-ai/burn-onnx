// Import the shared macro
use crate::include_models;
include_models!(shrink);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    use crate::backend::TestBackend;

    #[test]
    fn shrink() {
        let model: shrink::Model<TestBackend> = shrink::Model::default();

        let device = Default::default();
        let input = Tensor::<TestBackend, 2>::from_floats(
            [[-2.0, -1.0, 0.0, 1.0, 2.0], [-3.0, -2.5, -0.5, 0.5, 3.0]],
            &device,
        );
        let (out_no_bias, out_with_bias) = model.forward(input.clone(), input);

        let expected_no_bias = Tensor::<TestBackend, 2>::from_floats(
            [[-2.0, 0.0, 0.0, 0.0, 2.0], [-3.0, -2.5, 0.0, 0.0, 3.0]],
            &device,
        );
        let expected_with_bias = Tensor::<TestBackend, 2>::from_floats(
            [[-0.5, 0.0, 0.0, 0.0, 0.5], [-1.5, -1.0, 0.0, 0.0, 1.5]],
            &device,
        );

        assert_eq!(out_no_bias.into_data(), expected_no_bias.into_data());
        assert_eq!(out_with_bias.into_data(), expected_with_bias.into_data());
    }
}
