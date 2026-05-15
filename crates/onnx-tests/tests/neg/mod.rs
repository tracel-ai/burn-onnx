// Import the shared macro
use crate::include_models;
include_models!(neg);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn neg() {
        let device = Default::default();
        let model: neg::Model = neg::Model::new(&device);

        let input1 = Tensor::<4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let input2 = 99f64;

        let (output1, output2) = model.forward(input1, input2);
        let expected1 = TensorData::from([[[[-1.0f32, -4.0, -9.0, -25.0]]]]);
        let expected2 = -99f64;

        output1
            .to_data()
            .assert_approx_eq::<f32>(&expected1, Tolerance::default());

        assert_eq!(output2, expected2);
    }
}
