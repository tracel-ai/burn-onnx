use crate::include_models;
include_models!(
    scatter_nd_bool_reductions,
    scatter_nd,
    scatter_nd_2d,
    scatter_nd_add,
    scatter_nd_mul,
    scatter_nd_max,
    scatter_nd_min,
    scatter_nd_bool,
    scatter_nd_neg_idx
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Device, Tensor, TensorData};

    #[test]
    fn scatter_nd_1d() {
        let model: scatter_nd::Model = scatter_nd::Model::default();
        let device = Default::default();

        let data = Tensor::<1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 11., 3., 10., 9., 6., 7., 12.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_2d_slice() {
        let model: scatter_nd_2d::Model = scatter_nd_2d::Model::default();
        let device = Default::default();

        let data = Tensor::<2>::from_floats(
            [
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
            ],
            &device,
        );
        let updates = Tensor::<2>::from_floats([[5., 5., 5., 5.], [6., 6., 6., 6.]], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([
            [5f32, 5., 5., 5.],
            [1., 1., 1., 1.],
            [6., 6., 6., 6.],
            [1., 1., 1., 1.],
        ]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_add_reduction() {
        let model: scatter_nd_add::Model = scatter_nd_add::Model::default();
        let device = Default::default();

        let data = Tensor::<1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 13., 3., 14., 14., 6., 7., 20.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_mul_reduction() {
        let model: scatter_nd_mul::Model = scatter_nd_mul::Model::default();
        let device = Default::default();

        let data = Tensor::<1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 22., 3., 40., 45., 6., 7., 96.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_max_reduction() {
        let model: scatter_nd_max::Model = scatter_nd_max::Model::default();
        let device = Default::default();

        let data = Tensor::<1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 11., 3., 10., 9., 6., 7., 12.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_min_reduction() {
        let model: scatter_nd_min::Model = scatter_nd_min::Model::default();
        let device = Default::default();

        let data = Tensor::<1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 2., 3., 4., 5., 6., 7., 8.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_negative_indices() {
        // -1 means last; -3 means third-from-last. Exercises on-device normalization.
        let model: scatter_nd_neg_idx::Model = scatter_nd_neg_idx::Model::default();
        let device = Default::default();

        let data = Tensor::<1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<1>::from_floats([99., 88., 77.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 77., 3., 4., 5., 88., 7., 99.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_bool_none() {
        let model: scatter_nd_bool::Model = scatter_nd_bool::Model::default();
        let device = Default::default();

        let data = Tensor::<1, Bool>::from_bool(
            TensorData::from([false, false, false, false, false, false]),
            &device,
        );
        let updates = Tensor::<1, Bool>::from_bool(TensorData::from([true, true, true]), &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([false, true, false, true, false, true]);
        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn scatter_nd_bool_reductions() {
        // Expected values from scatter_nd_bool_reductions.py (ONNX reference, checked against
        // a NumPy loop and ONNX Runtime for add/mul). Each updated position differs from its
        // update, so or, and and assign all disagree. The indices are unique: burn documents
        // duplicate targets as undefined for scatter_nd reductions. Index 3 is an untouched true.
        let device = Default::default();
        let model = scatter_nd_bool_reductions::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/scatter_nd_bool_reductions.bpk"),
            &device,
        );
        let data = burn::tensor::Tensor::<1, burn::tensor::Bool>::from_bool(
            TensorData::from([false, true, true, true]),
            &device,
        );
        let indices = burn::tensor::Tensor::<2, burn::tensor::Int>::from_data(
            TensorData::from([[0i64], [1], [2]]),
            &device,
        );
        let updates = burn::tensor::Tensor::<1, burn::tensor::Bool>::from_bool(
            TensorData::from([true, false, true]),
            &device,
        );

        let (y_add, y_mul, y_max, y_min) = model.forward(data, indices, updates);

        let or = TensorData::from([true, true, true, true]);
        let and = TensorData::from([false, false, true, true]);
        y_add.to_data().assert_eq(&or, false);
        y_max.to_data().assert_eq(&or, false);
        y_mul.to_data().assert_eq(&and, false);
        y_min.to_data().assert_eq(&and, false);
    }
}
