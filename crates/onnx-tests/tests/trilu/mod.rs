use crate::include_models;
include_models!(trilu_lower, trilu_runtime_k, trilu_upper);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn trilu_upper() {
        let device = Default::default();
        let model: trilu_upper::Model = trilu_upper::Model::new(&device);
        let input = Tensor::<3>::from_floats([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]], &device);
        let expected = TensorData::from([[
            [1.0_f32, 2.0_f32, 3.0_f32],
            [0.0_f32, 5.0_f32, 6.0_f32],
            [0.0_f32, 0.0_f32, 9.0_f32],
        ]]);

        let output = model.forward(input).to_data();

        output.assert_eq(&expected, true);
    }

    #[test]
    fn trilu_lower() {
        let device = Default::default();
        let model: trilu_lower::Model = trilu_lower::Model::new(&device);
        let input = Tensor::<3>::from_floats([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]], &device);
        let expected = TensorData::from([[
            [1.0_f32, 0.0_f32, 0.0_f32],
            [4.0_f32, 5.0_f32, 0.0_f32],
            [7.0_f32, 8.0_f32, 9.0_f32],
        ]]);

        let output = model.forward(input).to_data();

        output.assert_eq(&expected, true);
    }

    #[test]
    fn trilu_runtime_k() {
        let device = Default::default();
        let model: trilu_runtime_k::Model = trilu_runtime_k::Model::new(&device);
        let input = || {
            Tensor::<2>::from_floats(
                [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
                &device,
            )
        };

        let below = model.forward(input(), -1).to_data();
        below.assert_eq(
            &TensorData::from([
                [0.0_f32, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [9.0, 10.0, 0.0, 0.0],
            ]),
            true,
        );

        let above = model.forward(input(), 2).to_data();
        above.assert_eq(
            &TensorData::from([
                [1.0_f32, 2.0, 3.0, 0.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]),
            true,
        );
    }
}
