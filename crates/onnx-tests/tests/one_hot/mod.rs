use crate::include_models;
include_models!(one_hot, one_hot_axis0, one_hot_float_values, one_hot_2d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    #[ignore = "burn-tensor one_hot_fill uses backend-default int indices that fail burn-flex int_scatter_add I64 contract - tracel-ai/burn#4776"]
    fn one_hot() {
        let device = Default::default();
        let model = one_hot::Model::<TestBackend>::new(&device);
        // ONNX OneHot's `indices` input is int64 per spec; pin the dtype so the
        // generated model receives an I64 tensor and the output carries the
        // ONNX-specified dtype. The assertion uses strict `assert_eq` against
        // an I64 TensorData literal (not `assert_approx_eq`) so it enforces the
        // output dtype, not just the element values. This is the only test that
        // covers the Int->Int one_hot_fill branch where burn-onnx must emit a
        // `.cast(DType::I64)` on top of `one_hot_fill` (which otherwise returns
        // the backend default int).
        let input: Tensor<TestBackend, 1, Int> =
            Tensor::from_data(TensorData::from([1i64, 0, 2]), (&device, DType::I64));
        let output: Tensor<TestBackend, 2, Int> = model.forward(input);
        let expected = TensorData::from([[0i64, 1, 0], [1, 0, 0], [0, 0, 1]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    #[ignore = "burn-tensor one_hot_fill uses backend-default int indices that fail burn-flex int_scatter_add I64 contract - tracel-ai/burn#4776"]
    fn one_hot_axis0() {
        // axis=0: one-hot dim inserted at front, depth=4, indices [0, 2, 3]
        // output shape [4, 3]
        let device = Default::default();
        let model = one_hot_axis0::Model::<TestBackend>::new(&device);

        // ONNX OneHot's `indices` input is int64 per spec; construct the test
        // tensor with explicit I64 dtype rather than relying on the backend's
        // default Int element.
        let input: Tensor<TestBackend, 1, Int> =
            Tensor::from_data(TensorData::from([0i64, 2, 3]), (&device, DType::I64));
        let output: Tensor<TestBackend, 2, Int> = model.forward(input);

        let expected = TensorData::from([[1i64, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    #[ignore = "burn-tensor one_hot_fill uses backend-default int indices that fail burn-flex int_scatter_add I64 contract - tracel-ai/burn#4776"]
    fn one_hot_float_values() {
        // Int indices with float values [0.0, 5.0] -> Float output
        let device = Default::default();
        let model = one_hot_float_values::Model::<TestBackend>::new(&device);

        let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints([0, 1, 2, 1], &device);
        let output: Tensor<TestBackend, 2> = model.forward(input);

        let expected = TensorData::from([
            [5.0f32, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    #[ignore = "burn-tensor one_hot_fill uses backend-default int indices that fail burn-flex int_scatter_add I64 contract - tracel-ai/burn#4776"]
    fn one_hot_2d() {
        // 2D input indices [2, 3] -> output [2, 3, 4]
        let device = Default::default();
        let model = one_hot_2d::Model::<TestBackend>::new(&device);

        // ONNX OneHot's `indices` input is int64 per spec; pass the dtype
        // explicitly so the model sees I64 rather than the backend-default
        // int that `from_data(data, &device)` would produce.
        let input: Tensor<TestBackend, 2, Int> = Tensor::from_data(
            TensorData::from([[0i64, 1, 2], [3, 0, 1]]),
            (&device, DType::I64),
        );
        let output: Tensor<TestBackend, 3, Int> = model.forward(input);

        let expected = TensorData::from([
            [[1i64, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}
