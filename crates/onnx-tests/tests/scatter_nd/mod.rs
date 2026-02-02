use crate::include_models;
include_models!(
    scatter_nd,
    scatter_nd_2d,
    scatter_nd_add,
    scatter_nd_mul,
    scatter_nd_max,
    scatter_nd_min,
    scatter_nd_bool
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn scatter_nd_1d() {
        let model: scatter_nd::Model<TestBackend> = scatter_nd::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<TestBackend, 1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 11., 3., 10., 9., 6., 7., 12.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_2d_slice() {
        let model: scatter_nd_2d::Model<TestBackend> = scatter_nd_2d::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 2>::from_floats(
            [
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
            ],
            &device,
        );
        let updates =
            Tensor::<TestBackend, 2>::from_floats([[5., 5., 5., 5.], [6., 6., 6., 6.]], &device);
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
        let model: scatter_nd_add::Model<TestBackend> = scatter_nd_add::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<TestBackend, 1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 13., 3., 14., 14., 6., 7., 20.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_mul_reduction() {
        let model: scatter_nd_mul::Model<TestBackend> = scatter_nd_mul::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<TestBackend, 1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 22., 3., 40., 45., 6., 7., 96.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_max_reduction() {
        let model: scatter_nd_max::Model<TestBackend> = scatter_nd_max::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<TestBackend, 1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 11., 3., 10., 9., 6., 7., 12.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_min_reduction() {
        let model: scatter_nd_min::Model<TestBackend> = scatter_nd_min::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 1>::from_floats([1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let updates = Tensor::<TestBackend, 1>::from_floats([9., 10., 11., 12.], &device);
        let output = model.forward(data, updates);

        let expected = TensorData::from([1f32, 2., 3., 4., 5., 6., 7., 8.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn scatter_nd_bool_none() {
        let model: scatter_nd_bool::Model<TestBackend> = scatter_nd_bool::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 1, Bool>::from_bool(
            TensorData::from([false, false, false, false, false, false]),
            &device,
        );
        let updates = Tensor::<TestBackend, 1, Bool>::from_bool(
            TensorData::from([true, true, true]),
            &device,
        );
        let output = model.forward(data, updates);

        let expected = TensorData::from([false, true, false, true, false, true]);
        assert_eq!(output.to_data(), expected);
    }
}
