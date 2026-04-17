// Include the models for this node type
use crate::include_models;
include_models!(scaler);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn test_scaler() {
        // Initialize the model
        let model: scaler::Model<TestBackend> = scaler::Model::default();

        let device = Default::default();

        // Input: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        // Formula: Y = (X - offset) * scale
        // With scale=2.0 and offset=1.0: Y = (X - 1.0) * 2.0
        let input =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

        let output = model.forward(input);

        // Expected: [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]
        let expected = TensorData::from([[0.0f32, 2.0, 4.0], [6.0, 8.0, 10.0]]);

        output.to_data().assert_eq(&expected, true);
    }
}
