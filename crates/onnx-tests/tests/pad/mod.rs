use crate::include_models;
include_models!(
    pad,
    pad_reflect,
    pad_edge,
    pad_runtime_constant,
    pad_optional_constant_value,
    pad_ndim
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn pad_constant() {
        let device = Default::default();
        let model: pad::Model<TestBackend> = pad::Model::new(&device);

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2.], [3., 4.], [5., 6.]], &device);
        let output = model.forward(input).to_data();
        let expected = TensorData::from([
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 1., 2., 0., 0., 0., 0.],
            [0.0_f32, 0., 3., 4., 0., 0., 0., 0.],
            [0.0_f32, 0., 5., 6., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_reflect_mode() {
        let device = Default::default();
        let model: pad_reflect::Model<TestBackend> = pad_reflect::Model::new(&device);

        // Input: 3x3 tensor
        let input = Tensor::<TestBackend, 2>::from_floats(
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            &device,
        );
        let output = model.forward(input).to_data();

        // Expected with reflect padding (1,1,1,1):
        // Reflect excludes the edge value when mirroring
        let expected = TensorData::from([
            [5.0_f32, 4., 5., 6., 5.],
            [2.0_f32, 1., 2., 3., 2.],
            [5.0_f32, 4., 5., 6., 5.],
            [8.0_f32, 7., 8., 9., 8.],
            [5.0_f32, 4., 5., 6., 5.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_runtime_constant_value() {
        let device = Default::default();
        let model: pad_runtime_constant::Model<TestBackend> =
            pad_runtime_constant::Model::new(&device);

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2.], [3., 4.]], &device);
        let output = model.forward(input, 5.0_f32).to_data();
        let expected = TensorData::from([
            [5.0_f32, 5., 5., 5.],
            [5.0_f32, 1., 2., 5.],
            [5.0_f32, 3., 4., 5.],
            [5.0_f32, 5., 5., 5.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_optional_constant_value() {
        let device = Default::default();
        let model: pad_optional_constant_value::Model<TestBackend> =
            pad_optional_constant_value::Model::new(&device);

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input).to_data();
        let expected = TensorData::from([
            [0.0_f32, 0., 0., 0., 0.],
            [0.0_f32, 1., 2., 3., 0.],
            [0.0_f32, 4., 5., 6., 0.],
            [0.0_f32, 0., 0., 0., 0.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_ndim() {
        let device = Default::default();
        let model: pad_ndim::Model<TestBackend> = pad_ndim::Model::new(&device);

        // Input: [1, 2, 3, 3] tensor with values 1..18
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]],
            ]]),
            &device,
        );
        let output = model.forward(input).to_data();

        // Pads: batch (1,0), channel (0,1), height (1,1), width (2,2)
        // Output shape: [2, 3, 5, 7]
        let expected = TensorData::from([
            [
                [
                    [0.0_f32, 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
            ],
            [
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 2., 3., 0., 0.],
                    [0., 0., 4., 5., 6., 0., 0.],
                    [0., 0., 7., 8., 9., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 10., 11., 12., 0., 0.],
                    [0., 0., 13., 14., 15., 0., 0.],
                    [0., 0., 16., 17., 18., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
            ],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_edge_mode() {
        let device = Default::default();
        let model: pad_edge::Model<TestBackend> = pad_edge::Model::new(&device);

        // Input: 2x3 tensor
        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input).to_data();

        // Expected with edge padding (1,1,1,1):
        // Edge replicates the boundary values
        let expected = TensorData::from([
            [1.0_f32, 1., 2., 3., 3.],
            [1.0_f32, 1., 2., 3., 3.],
            [4.0_f32, 4., 5., 6., 6.],
            [4.0_f32, 4., 5., 6., 6.],
        ]);

        output.assert_eq(&expected, true);
    }
}
