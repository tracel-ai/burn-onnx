use crate::include_models;
include_models!(
    scatter_elements,
    scatter_elements_axis1,
    scatter_elements_add,
    scatter_elements_add_partial,
    scatter_elements_mul,
    scatter_elements_max,
    scatter_elements_min,
    scatter_elements_bool,
    scatter_elements_3d,
    scatter_elements_1d,
    scatter_elements_int,
    scatter_opset10
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

    #[test]
    fn scatter_elements_default() {
        let device = Default::default();
        let model: scatter_elements::Model = scatter_elements::Model::new(&device);

        let data = Tensor::<2>::zeros([3, 3], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[2.0f32, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_opset10() {
        let device = Default::default();
        let model: scatter_opset10::Model = scatter_opset10::Model::new(&device);

        let data = Tensor::<2>::zeros([3, 3], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[2.0f32, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_axis1() {
        let device = Default::default();
        let model: scatter_elements_axis1::Model = scatter_elements_axis1::Model::new(&device);

        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        let indices = Tensor::<2, Int>::from_ints([[2, 0], [1, 2], [0, 1]], &device);
        let updates = Tensor::<2>::from_floats([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], &device);

        let output = model.forward(data, indices, updates);

        let expected =
            TensorData::from([[20.0f32, 2.0, 10.0], [4.0, 30.0, 40.0], [50.0, 60.0, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    fn make_test_inputs(device: &Device) -> (Tensor<2>, Tensor<2, Int>, Tensor<2>) {
        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], device);
        let updates = Tensor::<2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], device);
        (data, indices, updates)
    }

    // Updates that beat the existing value at some targets and fall below it at others, so
    // both branches of the max/min reduction are exercised.
    fn make_mixed_test_inputs(device: &Device) -> (Tensor<2>, Tensor<2, Int>, Tensor<2>) {
        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], device);
        let updates = Tensor::<2>::from_floats([[9.5, 1.1, 2.5], [0.5, 8.5, 7.5]], device);
        (data, indices, updates)
    }

    #[test]
    fn scatter_elements_with_add_reduction() {
        let device = Default::default();
        let model: scatter_elements_add::Model = scatter_elements_add::Model::new(&device);

        let data = Tensor::<2>::ones([3, 3], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[3.0f32, 2.1, 1.0], [2.0, 1.0, 3.2], [1.0, 3.1, 2.2]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_add_partial_indices() {
        let device = Default::default();
        let model: scatter_elements_add_partial::Model =
            scatter_elements_add_partial::Model::new(&device);

        // Indices cover only the first two columns; the third passes through.
        let data = Tensor::<2>::ones([3, 3], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0], [2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[1.0f32, 3.0, 1.0], [2.0, 5.0, 1.0], [4.0, 1.0, 1.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_mul_reduction() {
        let device = Default::default();
        let model: scatter_elements_mul::Model = scatter_elements_mul::Model::new(&device);

        let (data, indices, updates) = make_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // data[1,0]*=1.0=4, data[0,1]*=1.1=2.2, data[2,2]*=1.2=10.8
        // data[0,0]*=2.0=2, data[2,1]*=2.1=16.8, data[1,2]*=2.2=13.2
        let expected = TensorData::from([[2.0f32, 2.2, 3.0], [4.0, 5.0, 13.2], [7.0, 16.8, 10.8]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_max_reduction() {
        let device = Default::default();
        let model: scatter_elements_max::Model = scatter_elements_max::Model::new(&device);

        let (data, indices, updates) = make_mixed_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // data[1,0]=max(4,9.5)=9.5, data[0,1]=max(2,1.1)=2, data[2,2]=max(9,2.5)=9
        // data[0,0]=max(1,0.5)=1, data[2,1]=max(8,8.5)=8.5, data[1,2]=max(6,7.5)=7.5
        let expected = TensorData::from([[1.0f32, 2.0, 3.0], [9.5, 5.0, 7.5], [7.0, 8.5, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_min_reduction() {
        let device = Default::default();
        let model: scatter_elements_min::Model = scatter_elements_min::Model::new(&device);

        let (data, indices, updates) = make_mixed_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // data[1,0]=min(4,9.5)=4, data[0,1]=min(2,1.1)=1.1, data[2,2]=min(9,2.5)=2.5
        // data[0,0]=min(1,0.5)=0.5, data[2,1]=min(8,8.5)=8, data[1,2]=min(6,7.5)=6
        let expected = TensorData::from([[0.5f32, 1.1, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 2.5]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_negative_indices() {
        let device = Default::default();
        let model: scatter_elements::Model = scatter_elements::Model::new(&device);

        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        // ONNX allows indices down to -dim_size; -1 is the last row.
        let indices = Tensor::<2, Int>::from_ints([[-2, -3, -1], [-3, -1, -2]], &device);
        let updates = Tensor::<2>::from_floats([[9.5, 1.1, 2.5], [0.5, 8.5, 7.5]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[0.5f32, 1.1, 3.0], [9.5, 5.0, 7.5], [7.0, 8.5, 2.5]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_max_negative_indices() {
        let device = Default::default();
        let model: scatter_elements_max::Model = scatter_elements_max::Model::new(&device);

        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        let indices = Tensor::<2, Int>::from_ints([[-2, -3, -1], [-3, -1, -2]], &device);
        let updates = Tensor::<2>::from_floats([[9.5, 1.1, 2.5], [0.5, 8.5, 7.5]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[1.0f32, 2.0, 3.0], [9.5, 5.0, 7.5], [7.0, 8.5, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_max_with_infinity() {
        let device = Default::default();
        let model: scatter_elements_max::Model = scatter_elements_max::Model::new(&device);

        // max(-inf, 0.5) is 0.5. This guards a rejected implementation: folding max as a
        // scatter-add of `(updates - gathered).clamp_min(0)` computes -inf + inf here and
        // yields NaN. Assignment expressed as an add cannot be rescued for an infinite
        // existing value, whatever the delta is clamped to.
        let data = Tensor::<2>::from_floats(
            [
                [f32::NEG_INFINITY, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            &device,
        );
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[9.5, 1.1, 2.5], [0.5, 8.5, 7.5]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[0.5f32, 2.0, 3.0], [9.5, 5.0, 7.5], [7.0, 8.5, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_max_with_duplicate_indices() {
        let device = Default::default();
        let model: scatter_elements_max::Model = scatter_elements_max::Model::new(&device);

        // Every update targets row 0. ONNX folds them with max, so the result is the
        // largest update, not their sum. The larger update comes first so that the
        // expected value is not also the last write, which keeps the test able to tell a
        // correct fold from last-write-wins.
        //
        // burn documents duplicate indices as undefined for scatter_nd with Assign, Mul,
        // Min and Max. They fold sequentially on the CPU backends this suite runs on, but
        // cubecl races, so this assertion is CPU-backend only.
        let data = Tensor::<2>::zeros([3, 3], &device);
        let indices = Tensor::<2, Int>::from_ints([[0, 0, 0], [0, 0, 0]], &device);
        let updates = Tensor::<2>::from_floats([[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[3.0f32, 3.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_mul_negative_indices() {
        let device = Default::default();
        let model: scatter_elements_mul::Model = scatter_elements_mul::Model::new(&device);

        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        let indices = Tensor::<2, Int>::from_ints([[-2, -3, -1], [-3, -1, -2]], &device);
        let updates = Tensor::<2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], &device);

        let output = model.forward(data, indices, updates);

        // data[1,0]*=1.0=4, data[0,1]*=1.1=2.2, data[2,2]*=1.2=10.8
        // data[0,0]*=2.0=2, data[2,1]*=2.1=16.8, data[1,2]*=2.2=13.2
        let expected = TensorData::from([[2.0f32, 2.2, 3.0], [4.0, 5.0, 13.2], [7.0, 16.8, 10.8]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_3d_max_middle_axis() {
        let device = Default::default();
        let model: scatter_elements_3d::Model = scatter_elements_3d::Model::new(&device);

        // Rank 3 with axis=1, so the generated coordinate columns cover a non-unit outer
        // stride and the unit inner stride. Indices mix negative and positive values, and no
        // column repeats a target row: burn leaves duplicate indices undefined for scatter_nd.
        let data = Tensor::<3>::from_floats(
            [
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                ],
                [
                    [12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                ],
            ],
            &device,
        );
        let indices = Tensor::<3, Int>::from_ints(
            [
                [[0, -1, 1, -2], [2, 1, -3, 0]],
                [[-2, 0, 2, 1], [2, -1, 0, -3]],
            ],
            &device,
        );
        let updates = Tensor::<3>::from_floats(
            [
                [[0.0, 1.5, 3.0, 4.5], [6.0, 7.5, 9.0, 10.5]],
                [[12.0, 13.5, 15.0, 16.5], [18.0, 19.5, 21.0, 22.5]],
            ],
            &device,
        );

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([
            [
                [0.0f32, 1.0, 9.0, 10.5],
                [4.0, 7.5, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            [
                [12.0, 13.5, 21.0, 22.5],
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
        ]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_1d() {
        let device = Default::default();
        let model: scatter_elements_1d::Model = scatter_elements_1d::Model::new(&device);

        // Rank 1: the generated stride loop is empty and the scatter axis is the only
        // coordinate column.
        let data = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let indices = Tensor::<1, Int>::from_ints([2, 0, -1], &device);
        let updates = Tensor::<1>::from_floats([9.0, 8.0, 7.0], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([8.0f32, 2.0, 9.0, 7.0]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_int_dtype() {
        let device = Default::default();
        let model: scatter_elements_int::Model = scatter_elements_int::Model::new(&device);

        let data = Tensor::<2, Int>::from_ints([[1, 2, 3], [4, 5, 6], [7, 8, 9]], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, -2]], &device);
        let updates = Tensor::<2, Int>::from_ints([[10, 20, 30], [40, 50, 60]], &device);

        let output = model.forward(data, indices, updates);

        // `Tensor<D, Int>` graph inputs carry the backend's default int dtype, which is not
        // the same on every backend, so compare at a fixed width.
        let expected = TensorData::from([[40i64, 20, 3], [10, 5, 60], [7, 50, 30]]);
        assert_eq!(output.to_data().convert::<i64>(), expected);
    }

    #[test]
    fn scatter_elements_empty_indices() {
        let device = Default::default();
        let model: scatter_elements_max::Model = scatter_elements_max::Model::new(&device);

        // An empty index tensor is a legal ONNX no-op. It has to be special-cased because
        // scatter_nd rejects empty indices, and reshape reads a 0 in the target shape as
        // "keep the source dim" rather than as an empty dimension.
        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        let indices = Tensor::<2, Int>::zeros([0, 3], &device);
        let updates = Tensor::<2>::zeros([0, 3], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_bool_none() {
        let device = Default::default();
        let model: scatter_elements_bool::Model = scatter_elements_bool::Model::new(&device);

        let data = Tensor::<2, Bool>::from_bool(
            TensorData::from([[true, true, true], [false, false, false]]),
            &device,
        );
        let indices = Tensor::<2, Int>::from_ints([[2, 0, 1], [1, 2, 0]], &device);
        let updates = Tensor::<2, Bool>::from_bool(
            TensorData::from([[false, false, true], [true, true, false]]),
            &device,
        );

        let output = model.forward(data, indices, updates);

        // Targets are both set and cleared, which a logical-or scatter could not express.
        let expected = TensorData::from([[false, true, false], [false, true, true]]);
        output.to_data().assert_eq(&expected, false);
    }

    // Indices narrower than data on a non-axis dimension take the scatter_nd path.

    #[test]
    fn scatter_elements_partial_indices() {
        let device = Default::default();
        let model: scatter_elements::Model = scatter_elements::Model::new(&device);

        let data = Tensor::<2>::zeros([3, 3], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0], [2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[0.0f32, 2.0, 0.0], [1.0, 4.0, 0.0], [3.0, 0.0, 0.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_max_partial_indices() {
        let device = Default::default();
        let model: scatter_elements_max::Model = scatter_elements_max::Model::new(&device);

        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0], [2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[9.5, 1.5], [3.0, 8.0]], &device);

        let output = model.forward(data, indices, updates);

        // [1,0]=max(4,9.5), [0,1]=max(2,1.5), [2,0]=max(7,3), [1,1]=max(5,8)
        let expected = TensorData::from([[1.0f32, 2.0, 3.0], [9.5, 8.0, 6.0], [7.0, 8.0, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_bool_partial_indices() {
        let device = Default::default();
        let model: scatter_elements_bool::Model = scatter_elements_bool::Model::new(&device);

        let data = Tensor::<2, Bool>::from_bool(
            TensorData::from([[true, true, true], [false, false, false]]),
            &device,
        );
        // Axis 1; only the first row is scattered.
        let indices = Tensor::<2, Int>::from_ints([[2, 0, 1]], &device);
        let updates =
            Tensor::<2, Bool>::from_bool(TensorData::from([[false, false, true]]), &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[false, true, false], [false, false, false]]);
        output.to_data().assert_eq(&expected, false);
    }
}
