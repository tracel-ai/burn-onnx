use crate::include_models;
include_models!(one_hot, one_hot_axis0, one_hot_float_values, one_hot_2d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn one_hot() {
        let device = Default::default();
        let model = one_hot::Model::<TestBackend>::new(&device);
        let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints([1, 0, 2], &device);
        let expected: Tensor<TestBackend, 2, Int> =
            Tensor::from_data(TensorData::from([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), &device);
        let output: Tensor<TestBackend, 2, Int> = model.forward(input);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn one_hot_axis0() {
        // axis=0: one-hot dim inserted at front, depth=4, indices [0, 2, 3]
        // output shape [4, 3]
        let device = Default::default();
        let model = one_hot_axis0::Model::<TestBackend>::new(&device);

        let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints([0, 2, 3], &device);
        let output: Tensor<TestBackend, 2, Int> = model.forward(input);

        let expected = TensorData::from([[1i64, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
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
    fn one_hot_2d() {
        // 2D input indices [2, 3] -> output [2, 3, 4]
        let device = Default::default();
        let model = one_hot_2d::Model::<TestBackend>::new(&device);

        let input: Tensor<TestBackend, 2, Int> =
            Tensor::from_data(TensorData::from([[0i64, 1, 2], [3, 0, 1]]), &device);
        let output: Tensor<TestBackend, 3, Int> = model.forward(input);

        let expected = TensorData::from([
            [[1i64, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}
