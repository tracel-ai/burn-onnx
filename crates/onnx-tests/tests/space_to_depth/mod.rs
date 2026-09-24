// Import the shared macro
use crate::include_models;
include_models!(space_to_depth, space_to_depth_multi, space_to_depth_int);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn space_to_depth() {
        let device = Default::default();
        let model: space_to_depth::Model = space_to_depth::Model::new(&device);

        let input = Tensor::<4>::from_floats(
            [
                [[
                    [0.5, -0.14, 0.65, 1.52, -0.23, -0.23],
                    [1.58, 0.77, -0.47, 0.54, -0.46, -0.47],
                    [0.24, -1.91, -1.72, -0.56, -1.01, 0.31],
                    [-0.91, -1.41, 1.47, -0.23, 0.07, -1.42],
                ]],
                [[
                    [-0.54, 0.11, -1.15, 0.38, -0.6, -0.29],
                    [-0.6, 1.85, -0.01, -1.06, 0.82, -1.22],
                    [0.21, -1.96, -1.33, 0.2, 0.74, 0.17],
                    [-0.12, -0.3, -1.48, -0.72, -0.46, 1.06],
                ]],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [
                [[0.5, 0.65, -0.23], [0.24, -1.72, -1.01]],
                [[-0.14, 1.52, -0.23], [-1.91, -0.56, 0.31]],
                [[1.58, -0.47, -0.46], [-0.91, 1.47, 0.07]],
                [[0.77, 0.54, -0.47], [-1.41, -0.23, -1.42]],
            ],
            [
                [[-0.54, -1.15, -0.6], [0.21, -1.33, 0.74]],
                [[0.11, 0.38, -0.29], [-1.96, 0.2, 0.17]],
                [[-0.6, -0.01, 0.82], [-0.12, -1.48, -0.46]],
                [[1.85, -1.06, -1.22], [-0.3, -0.72, 1.06]],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn space_to_depth_multi_channel() {
        let device = Default::default();
        let model: space_to_depth_multi::Model = space_to_depth_multi::Model::new(&device);
        let input = Tensor::<1, burn::tensor::Int>::arange(0..16, &device)
            .float()
            .reshape([1, 2, 2, 4]);

        let output = model.forward(input);

        output.to_data().assert_eq(
            &TensorData::from([[
                [[0.0f32, 2.0]],
                [[8.0, 10.0]],
                [[1.0, 3.0]],
                [[9.0, 11.0]],
                [[4.0, 6.0]],
                [[12.0, 14.0]],
                [[5.0, 7.0]],
                [[13.0, 15.0]],
            ]]),
            true,
        );
    }

    #[test]
    fn space_to_depth_int() {
        let device = Default::default();
        let model: space_to_depth_int::Model = space_to_depth_int::Model::new(&device);
        let input =
            Tensor::<1, burn::tensor::Int>::arange(0..16, (&device, burn::tensor::DType::I64))
                .reshape([1, 2, 2, 4]);

        let output = model.forward(input);

        output.to_data().assert_eq(
            &TensorData::from([[
                [[0i64, 2]],
                [[8, 10]],
                [[1, 3]],
                [[9, 11]],
                [[4, 6]],
                [[12, 14]],
                [[5, 7]],
                [[13, 15]],
            ]]),
            true,
        );
    }
}
