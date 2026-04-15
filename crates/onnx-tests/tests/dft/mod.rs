use crate::include_models;
include_models!(dft_onesided, dft_full);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::TestBackend;

    #[test]
    fn dft_onesided() {
        let device = Default::default();
        let model: dft_onesided::Model<TestBackend> = dft_onesided::Model::new(&device);

        // Input: [1, 8, 1] real signal [1, 2, 3, 4, 5, 6, 7, 8]
        let input = burn::tensor::Tensor::<TestBackend, 3>::from_floats(
            [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]],
            &device,
        );

        let output = model.forward(input);

        // Expected onesided DFT output: [1, 5, 2]
        let expected = burn::tensor::Tensor::<TestBackend, 3>::from_floats(
            [[
                [36.0f32, 0.0],
                [-4.0, 9.656_855],
                [-4.0, 4.0],
                [-4.0, 1.656_854_3],
                [-4.0, 0.0],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn dft_full_spectrum() {
        let device = Default::default();
        let model: dft_full::Model<TestBackend> = dft_full::Model::new(&device);

        // Input: [1, 8, 1] real signal [1, 2, 3, 4, 5, 6, 7, 8]
        let input = burn::tensor::Tensor::<TestBackend, 3>::from_floats(
            [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]],
            &device,
        );

        let output = model.forward(input);

        // Expected full DFT output: [1, 8, 2]
        // Full spectrum = onesided + conjugate mirror
        let expected = burn::tensor::Tensor::<TestBackend, 3>::from_floats(
            [[
                [36.0f32, 0.0],
                [-4.0, 9.656_855],
                [-4.0, 4.0],
                [-4.0, 1.656_854_3],
                [-4.0, 0.0],
                [-4.0, -1.656_854_3],
                [-4.0, -4.0],
                [-4.0, -9.656_855],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
