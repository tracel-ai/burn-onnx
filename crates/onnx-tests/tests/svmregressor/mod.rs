// Import the shared macro
use crate::include_models;
include_models!(svmregressor, svmregressor_rbf);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn svmregressor_linear() {
        let device = Default::default();
        let model: svmregressor::Model<TestBackend> = svmregressor::Model::new(&device);

        // Input: [3, 2]
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414f32, -0.13826430],
                [0.64768857, 1.52302980],
                [-0.23415338, -0.23413695],
            ],
            &device,
        );

        let output = model.forward(input);
        // Output: [3] — one score per sample
        let expected =
            Tensor::<TestBackend, 1>::from_floats([0.25164291f32, 0.17615581, 0.61707670], &device);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn svmregressor_rbf() {
        let device = Default::default();
        let model: svmregressor_rbf::Model<TestBackend> = svmregressor_rbf::Model::new(&device);

        // Same input as the linear test (np.random.seed(42), shape [3, 2])
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414f32, -0.13826430],
                [0.64768857, 1.52302980],
                [-0.23415338, -0.23413695],
            ],
            &device,
        );

        let output = model.forward(input);
        // Expected from onnx.reference.ReferenceEvaluator (squeezed to [3])
        let expected =
            Tensor::<TestBackend, 1>::from_floats([0.23001021f32, 0.81986856, 0.21695474], &device);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }
}
