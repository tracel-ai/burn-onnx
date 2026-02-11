use crate::include_models;
include_models!(
    scatter_elements,
    scatter_elements_axis1,
    scatter_elements_add,
    scatter_elements_mul,
    scatter_elements_max,
    scatter_elements_min
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Int, Tensor, TensorData, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn scatter_elements_default() {
        let device = Default::default();
        let model: scatter_elements::Model<TestBackend> = scatter_elements::Model::new(&device);

        let data = Tensor::<TestBackend, 2>::zeros([3, 3], &device);
        let indices = Tensor::<TestBackend, 2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], &device);
        let updates =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[2.0f32, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_axis1() {
        let device = Default::default();
        let model: scatter_elements_axis1::Model<TestBackend> =
            scatter_elements_axis1::Model::new(&device);

        let data = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            &device,
        );
        let indices = Tensor::<TestBackend, 2, Int>::from_ints([[2, 0], [1, 2], [0, 1]], &device);
        let updates = Tensor::<TestBackend, 2>::from_floats(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            &device,
        );

        let output = model.forward(data, indices, updates);

        let expected =
            TensorData::from([[20.0f32, 2.0, 10.0], [4.0, 30.0, 40.0], [50.0, 60.0, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }

    fn make_test_inputs(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> (
        Tensor<TestBackend, 2>,
        Tensor<TestBackend, 2, Int>,
        Tensor<TestBackend, 2>,
    ) {
        let data = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            device,
        );
        let indices = Tensor::<TestBackend, 2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], device);
        let updates =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], device);
        (data, indices, updates)
    }

    #[test]
    fn scatter_elements_with_add_reduction() {
        let device = Default::default();
        let model: scatter_elements_add::Model<TestBackend> =
            scatter_elements_add::Model::new(&device);

        let data = Tensor::<TestBackend, 2>::ones([3, 3], &device);
        let indices = Tensor::<TestBackend, 2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], &device);
        let updates =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[3.0f32, 2.1, 1.0], [2.0, 1.0, 3.2], [1.0, 3.1, 2.2]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_mul_reduction() {
        let device = Default::default();
        let model: scatter_elements_mul::Model<TestBackend> =
            scatter_elements_mul::Model::new(&device);

        let (data, indices, updates) = make_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // data[1,0]*=1.0=4, data[0,1]*=1.1=2.2, data[2,2]*=1.2=10.8
        // data[0,0]*=2.0=2, data[2,1]*=2.1=16.8, data[1,2]*=2.2=13.2
        let expected = TensorData::from([[2.0f32, 2.2, 3.0], [4.0, 5.0, 13.2], [7.0, 16.8, 10.8]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_max_reduction() {
        let device = Default::default();
        let model: scatter_elements_max::Model<TestBackend> =
            scatter_elements_max::Model::new(&device);

        let (data, indices, updates) = make_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // All updates are smaller than existing data values at target positions
        let expected = TensorData::from([[2.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_min_reduction() {
        let device = Default::default();
        let model: scatter_elements_min::Model<TestBackend> =
            scatter_elements_min::Model::new(&device);

        let (data, indices, updates) = make_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // Updates are smaller than existing data at target positions
        let expected = TensorData::from([[1.0f32, 1.1, 3.0], [1.0, 5.0, 2.2], [7.0, 2.1, 1.2]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
