// Import the shared macro
use crate::include_models;
include_models!(col2im);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn col2im() {
        let device = Default::default();
        let model: col2im::Model<TestBackend> = col2im::Model::new(&device);

        // Input shape: [1, 5, 5] -> Col2Im -> Output shape: [1, 1, 5, 5]
        // image_shape=[5,5], block_shape=[1,5]
        // C = 5 / (1*5) = 1, output = [1, 1, 5, 5]
        //
        // With block_shape=[1,5], each column has 1*5=5 elements.
        // Sliding window positions: H: (5-1)/1+1=5, W: (5-5)/1+1=1 => L=5
        // Input [1,5,5]: channels=1, block_elements=5, positions=5
        //
        // The col2im operation places each position's 1x5 block at:
        //   pos 0: row 0, cols 0..4
        //   pos 1: row 1, cols 0..4
        //   ...
        //   pos 4: row 4, cols 0..4
        //
        // Each "column" i from input maps to position i (row i) in the output.
        // The 5 elements in column i become the 5 values across width for row i.
        //
        // So the output is: output[0,0,row,col] = input[0, col, row]
        // i.e., a transpose of the spatial dims.

        // Use known test values from the ONNX ReferenceEvaluator
        let input = Tensor::<TestBackend, 3>::from_floats(
            [[
                [0.5, -0.14, 0.65, 1.52, -0.23],
                [-0.23, 1.58, 0.77, -0.47, 0.54],
                [-0.46, -0.47, 0.24, -1.91, -1.72],
                [-0.56, -1.01, 0.31, -0.91, -1.41],
                [1.47, -0.23, 0.07, -1.42, -0.54],
            ]],
            &device,
        );

        let output = model.forward(input);

        // Expected output from ONNX ReferenceEvaluator
        let expected = TensorData::from([[[
            [0.5_f32, -0.23, -0.46, -0.56, 1.47],
            [-0.14, 1.58, -0.47, -1.01, -0.23],
            [0.65, 0.77, 0.24, 0.31, 0.07],
            [1.52, -0.47, -1.91, -0.91, -1.42],
            [-0.23, 0.54, -1.72, -1.41, -0.54],
        ]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
