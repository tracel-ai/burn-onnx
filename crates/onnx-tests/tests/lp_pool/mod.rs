// Import the shared macro
use crate::include_models;
include_models!(lp_pool1d, lp_pool2d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn lp_pool1d() {
        let device = Default::default();
        let model: lp_pool1d::Model<TestBackend> = lp_pool1d::Model::new(&device);

        let input =
            Tensor::<TestBackend, 3>::from_floats([[[-1.0, 2.0, -3.0, 4.0, -5.0]]], &device);
        let output = model.forward(input);

        let expected = TensorData::from([[[2.0800838f32, 4.6260653, 5.738794]]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(0.01, 0.001));
    }

    #[test]
    fn lp_pool2d() {
        let device = Default::default();
        let model: lp_pool2d::Model<TestBackend> = lp_pool2d::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, -2.0, 3.0, -4.0],
                [5.0, -6.0, 7.0, -8.0],
                [9.0, -10.0, 11.0, -12.0],
                [13.0, -14.0, 15.0, -16.0],
            ]]],
            &device,
        );
        let output = model.forward(input);

        let expected = TensorData::from([[[
            [8.124039f32, 13.341664],
            [15.556349, 22.671568],
            [23.366642, 32.280025],
            [19.104973, 26.019224],
        ]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::rel_abs(0.01, 0.001));
    }
}
