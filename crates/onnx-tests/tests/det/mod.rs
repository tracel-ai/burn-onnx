use crate::include_models;
include_models!(det, det_batched);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_2x2() {
        // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
        let device = Default::default();
        let model: det::Model = det::Model::new(&device);

        let input = burn::tensor::Tensor::<2>::from_floats([[1.0f32, 2.0], [3.0, 4.0]], &device);

        let output: f32 = model.forward(input);
        let expected = -2.0f32;

        let diff = (output - expected).abs();
        assert!(
            diff < 1e-4,
            "Expected det = {expected}, got {output}, diff = {diff}"
        );
    }

    #[test]
    fn det_batched_3x3() {
        // Two 3x3 matrices:
        // Matrix 0: [[1,2,3],[0,1,4],[5,6,0]] -> det = 1
        // Matrix 1: [[2,0,0],[0,3,0],[0,0,4]] -> det = 24
        let device = Default::default();
        let model: det_batched::Model = det_batched::Model::new(&device);

        let input = burn::tensor::Tensor::<3>::from_floats(
            [
                [[1.0f32, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
            ],
            &device,
        );

        let output = model.forward(input);
        let expected = burn::tensor::Tensor::<1>::from_floats([1.0f32, 24.0], &device);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
