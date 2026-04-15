// Import the shared macro
use crate::include_models;
include_models!(
    expand,
    expand_scalar,
    expand_tensor,
    expand_shape,
    expand_with_where_shape,
    expand_max_semantics,
    expand_dynamic_where,
    expand_shape_as_data
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Shape, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn expand() {
        let device = Default::default();
        let model: expand::Model<TestBackend> = expand::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[-1.0], [1.0]], &device);

        let output = model.forward(input1);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_tensor() {
        let device = Default::default();
        let model: expand_tensor::Model<TestBackend> = expand_tensor::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[-1.0], [1.0]], &device);
        let input2 = Tensor::<TestBackend, 1, Int>::from_ints([2, 2], &device);

        let output = model.forward(input1, input2);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_scalar() {
        let device = Default::default();
        let model: expand_scalar::Model<TestBackend> = expand_scalar::Model::new(&device);

        let input = 5i64;
        let shape = Tensor::<TestBackend, 1, Int>::from_ints([2, 2], &device);

        let output = model.forward(input, shape);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);

        // Verify values: all elements should be 5. The ONNX model's source constant is
        // int64, so the generated Expand output is I64. Compare via an explicit I64
        // TensorData rather than Tensor::from_ints (which defaults to the backend's int).
        let expected = TensorData::from([[5i64, 5], [5, 5]]);
        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn expand_shape() {
        let device = Default::default();
        let model: expand_shape::Model<TestBackend> = expand_shape::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[-1.0], [1.0], [1.0], [1.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::zeros([4, 4], &device);

        let output = model.forward(input1, input2);
        let expected_shape = Shape::from([4, 4]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_with_where_shape() {
        let device = Default::default();
        // Use Model::default() to load constants from the record file
        let model: expand_with_where_shape::Model<TestBackend> =
            expand_with_where_shape::Model::default();

        // Input tensor to be expanded
        let input = Tensor::<TestBackend, 3>::ones([1, 1, 4], &device);

        // The model doesn't actually take condition as input - it's built into the model
        let output = model.forward(input);

        // The model has two constant shapes [2,3,4] selected by Where, then used in Expand
        // Result should be expanded to shape [2, 3, 4]
        let expected_shape = Shape::from([2, 3, 4]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_dynamic_where() {
        // Tests Expand with a fully dynamic shape from a Where/Shape chain.
        // The shape tensor has no static_shape info, exercising the input-rank fallback.
        let device = Default::default();
        let model: expand_dynamic_where::Model<TestBackend> =
            expand_dynamic_where::Model::new(&device);

        let input_data = Tensor::<TestBackend, 1>::from_floats([10.0, 20.0, 30.0], &device);
        let input_flag = true;

        let output = model.forward(input_data.clone(), input_flag);

        // flag=true -> shape = Shape(input) = [3] -> Expand is identity
        let expected_shape = Shape::from([3]);
        assert_eq!(output.shape(), expected_shape);
        output.into_data().assert_eq(&input_data.into_data(), true);
    }

    #[test]
    fn expand_shape_as_data() {
        // Tests issue #266: Shape node output used as Expand's data input (input[0]).
        // Shape([3,4]) -> [3, 4] int64, Expand([3,4], [2,2]) -> [[3,4],[3,4]]
        let device = Default::default();
        let model: expand_shape_as_data::Model<TestBackend> =
            expand_shape_as_data::Model::new(&device);

        let input = Tensor::<TestBackend, 2>::zeros([3, 4], &device);
        let output = model.forward(input);

        assert_eq!(output.shape(), Shape::from([2, 2]));
        // Shape values in ONNX are int64; the generated Expand output preserves that dtype.
        let expected = TensorData::from([[3i64, 4], [3, 4]]);
        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn expand_max_semantics() {
        // Tests ONNX Expand's max-semantics behavior:
        // When shape_dim=1 but input_dim>1, ONNX keeps the input_dim (not replaces with 1)
        // Input: [2, 3], Shape: [1, 1], Expected Output: [2, 3]
        let device = Default::default();
        let model: expand_max_semantics::Model<TestBackend> =
            expand_max_semantics::Model::new(&device);

        let input =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

        let output = model.forward(input.clone());

        // ONNX max-semantics: output_dim = max(input_dim, shape_dim)
        // max(2, 1) = 2, max(3, 1) = 3 => [2, 3]
        let expected_shape = Shape::from([2, 3]);
        assert_eq!(output.shape(), expected_shape);

        // Data should be preserved (no actual broadcasting occurred)
        output.into_data().assert_eq(&input.into_data(), true);
    }
}
