// Import the shared macro
use crate::include_models;
include_models!(
    upsample_nearest_opset7,
    upsample_nearest_opset9,
    upsample_nearest_runtime_scales
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn upsample_nearest_opset7() {
        // Scales come from the opset 7 float-list attribute: [1, 1, 2, 2]
        let device = Default::default();
        let model: upsample_nearest_opset7::Model = upsample_nearest_opset7::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/upsample_nearest_opset7.bpk"),
            &device,
        );

        let input = Tensor::<4>::from_floats(
            [[
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ]],
            &device,
        );
        let output = model.forward(input);

        let expected = TensorData::from([[
            [
                [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
                [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            ],
            [
                [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
                [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
                [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
                [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
            ],
        ]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn upsample_nearest_opset9() {
        // Scales come from an initializer: [1, 1, 3, 2]
        let device = Default::default();
        let model: upsample_nearest_opset9::Model = upsample_nearest_opset9::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/upsample_nearest_opset9.bpk"),
            &device,
        );

        let input = Tensor::<4>::from_floats(
            [[
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ]],
            &device,
        );
        let output = model.forward(input);

        let expected = TensorData::from([[
            [
                [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
                [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
                [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            ],
            [
                [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
                [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
                [6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
                [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
                [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
                [9.0, 9.0, 10.0, 10.0, 11.0, 11.0],
            ],
        ]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn upsample_nearest_runtime_scales() {
        // Scales are a graph input, so the output size is computed at runtime
        let device = Default::default();
        let model: upsample_nearest_runtime_scales::Model =
            upsample_nearest_runtime_scales::Model::from_file(
                concat!(
                    env!("OUT_DIR"),
                    "/model/upsample_nearest_runtime_scales.bpk"
                ),
                &device,
            );

        let input = Tensor::<4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
        let scales = Tensor::<1>::from_floats([1.0, 1.0, 2.0, 3.0], &device);
        let output = model.forward(input, scales);

        let expected = TensorData::from([[[
            [1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
            [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        ]]]);

        output.to_data().assert_eq(&expected, true);
    }
}
