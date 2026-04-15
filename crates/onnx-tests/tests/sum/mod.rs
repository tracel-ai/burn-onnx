// Include the models for this node type
use crate::include_models;
include_models!(sum, sum_int);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn sum_tensor_and_tensor() {
        let device = Default::default();
        let model: sum::Model<TestBackend> = sum::Model::default();

        let input1 = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4.], &device);
        let input2 = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4.], &device);
        let input3 = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4.], &device);

        let output = model.forward(input1, input2, input3);
        let expected = TensorData::from([3f32, 6., 9., 12.]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sum_int_tensor_and_int_tensor() {
        let device = Default::default();
        let model: sum_int::Model<TestBackend> = sum_int::Model::default();

        // The ONNX model (sum_int.py) declares all three inputs and the
        // output as INT64. Construct the inputs with explicit I64 dtype so
        // the dtype-preserving Sum carries I64 through to the output,
        // rather than relying on the backend's default Int element.
        let input1 = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([1i64, 2, 3, 4]),
            (&device, DType::I64),
        );
        let input2 = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([1i64, 2, 3, 4]),
            (&device, DType::I64),
        );
        let input3 = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::from([1i64, 2, 3, 4]),
            (&device, DType::I64),
        );

        let output = model.forward(input1, input2, input3);
        let expected = TensorData::from([3i64, 6, 9, 12]);

        output.to_data().assert_eq(&expected, true);
    }
}
