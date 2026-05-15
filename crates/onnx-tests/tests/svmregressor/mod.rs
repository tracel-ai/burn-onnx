// Import the shared macro
use crate::include_models;
include_models!(
    svmregressor,
    svmregressor_rbf,
    svmregressor_poly,
    svmregressor_sigmoid,
    svmregressor_logistic,
    svmregressor_softmax_zero
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, Tolerance};

    /// np.random.seed(42) input, shape [3, 2]
    fn test_input(device: &Device) -> Tensor<2> {
        Tensor::<2>::from_floats(
            [
                [0.49671414f32, -0.13826430],
                [0.64768857, 1.52302980],
                [-0.23415338, -0.23413695],
            ],
            device,
        )
    }

    #[test]
    fn svmregressor_linear() {
        let device = Default::default();
        let model: svmregressor::Model = svmregressor::Model::new(&device);
        let output = model.forward(test_input(&device));
        let expected = Tensor::<1>::from_floats([0.25164291f32, 0.17615581, 0.61707670], &device);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn svmregressor_rbf() {
        let device = Default::default();
        let model: svmregressor_rbf::Model = svmregressor_rbf::Model::new(&device);
        let output = model.forward(test_input(&device));
        // Expected from onnx.reference.ReferenceEvaluator
        let expected = Tensor::<1>::from_floats([0.23001021f32, 0.81986856, 0.21695474], &device);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn svmregressor_poly() {
        let device = Default::default();
        let model: svmregressor_poly::Model = svmregressor_poly::Model::new(&device);
        let output = model.forward(test_input(&device));
        // Expected from onnx.reference.ReferenceEvaluator (gamma=1, coef0=1, degree=2)
        let expected = Tensor::<1>::from_floats([0.11270308f32, -18.28600883, 0.38438398], &device);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn svmregressor_sigmoid() {
        let device = Default::default();
        let model: svmregressor_sigmoid::Model = svmregressor_sigmoid::Model::new(&device);
        let output = model.forward(test_input(&device));
        // Expected from onnx.reference.ReferenceEvaluator (gamma=0.5, coef0=0.1)
        let expected = Tensor::<1>::from_floats([0.44991118f32, 0.96034056, 0.56224179], &device);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn svmregressor_logistic() {
        let device = Default::default();
        let model: svmregressor_logistic::Model = svmregressor_logistic::Model::new(&device);
        let output = model.forward(test_input(&device));
        // Expected: sigmoid(LINEAR output) = 1/(1+exp(-y))
        let expected = Tensor::<1>::from_floats([0.56258082f32, 0.54392546, 0.64955342], &device);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn svmregressor_softmax_zero() {
        let device = Default::default();
        let model: svmregressor_softmax_zero::Model =
            svmregressor_softmax_zero::Model::new(&device);
        let output = model.forward(test_input(&device));
        // SOFTMAX_ZERO appends an implicit zero class. For single-target output y
        // this is exp(y)/(exp(y)+1) = sigmoid(y), so values match LOGISTIC.
        let expected = Tensor::<1>::from_floats([0.56258088f32, 0.54392540, 0.64955342], &device);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }
}
