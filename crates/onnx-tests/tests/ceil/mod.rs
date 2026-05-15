use crate::include_models;
include_models!(ceil);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    #[test]
    fn ceil_test() {
        // Test for ceil
        let device = Default::default();
        let model = ceil::Model::new(&device);

        let input = Tensor::<1>::from_floats([-0.5, 1.5, 2.1], &device);
        let expected = Tensor::<1>::from_floats([0., 2., 3.], &device);

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
