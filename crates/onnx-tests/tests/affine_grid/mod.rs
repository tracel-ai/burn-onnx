use crate::include_models;
include_models!(affine_grid_2d, affine_grid_3d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData, Tolerance};

    fn theta_2d(device: &burn::tensor::Device) -> Tensor<3> {
        Tensor::from_floats(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.5, -0.25, 0.1], [0.3, 0.8, -0.2]],
            ],
            device,
        )
    }

    #[test]
    fn affine_grid_2d_runtime_size() {
        // Expected values from affine_grid_2d.py (ONNX reference, matches ONNX Runtime).
        // size is a graph input, as in the official tests: N=2, H=3, W=4.
        let device = Default::default();
        let model = affine_grid_2d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/affine_grid_2d.bpk"),
            &device,
        );
        let size = Tensor::<1, Int>::from_ints([2, 1, 3, 4], &device);

        let (grid, grid_align_corners) = model.forward(theta_2d(&device), size);

        let expected = TensorData::from(GRID);
        grid.to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-5));
        let expected = TensorData::from(GRID_ALIGN_CORNERS);
        grid_align_corners
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-5));
    }

    #[test]
    fn affine_grid_2d_single_row() {
        // H=1: without align_corners the only row is at 0; with align_corners a single
        // position is -1, as in the ONNX reference (ONNX Runtime gives 1, PyTorch 0).
        let device = Default::default();
        let model = affine_grid_2d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/affine_grid_2d.bpk"),
            &device,
        );
        let size = Tensor::<1, Int>::from_ints([2, 1, 1, 3], &device);

        let (grid, grid_align_corners) = model.forward(theta_2d(&device), size);

        let expected = TensorData::from(GRID_SINGLE_ROW);
        grid.to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-5));
        let expected = TensorData::from(GRID_SINGLE_ROW_ALIGN_CORNERS);
        grid_align_corners
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-5));
    }

    #[test]
    #[should_panic(expected = "AffineGrid: size must be non-negative")]
    fn affine_grid_2d_rejects_negative_size() {
        let device = Default::default();
        let model = affine_grid_2d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/affine_grid_2d.bpk"),
            &device,
        );
        let size = Tensor::<1, Int>::from_ints([2, 1, -3, 4], &device);

        let _ = model.forward(theta_2d(&device), size);
    }

    #[test]
    #[should_panic(expected = "AffineGrid: theta has batch 2 but size has N = 3")]
    fn affine_grid_2d_rejects_batch_mismatch() {
        let device = Default::default();
        let model = affine_grid_2d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/affine_grid_2d.bpk"),
            &device,
        );
        let size = Tensor::<1, Int>::from_ints([3, 1, 3, 4], &device);

        let _ = model.forward(theta_2d(&device), size);
    }

    #[test]
    fn affine_grid_3d_constant_size() {
        // Expected values from affine_grid_3d.py (ONNX reference, matches ONNX Runtime).
        // Constant size [1, 1, 2, 3, 2]: D=2, H=3, W=2.
        let device = Default::default();
        let model = affine_grid_3d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/affine_grid_3d.bpk"),
            &device,
        );
        let theta = Tensor::<3>::from_floats(
            [[
                [0.9, 0.1, 0.0, 0.2],
                [-0.1, 1.1, 0.3, 0.0],
                [0.0, 0.2, 0.7, -0.4],
            ]],
            &device,
        );

        let (grid, grid_align_corners) = model.forward(theta);

        let expected = TensorData::from(GRID_3D);
        grid.to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-5));
        let expected = TensorData::from(GRID_3D_ALIGN_CORNERS);
        grid_align_corners
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-5));
    }

    const GRID: [[[[f32; 2]; 4]; 3]; 2] = [
        [
            [
                [-0.75, -0.666667],
                [-0.25, -0.666667],
                [0.25, -0.666667],
                [0.75, -0.666667],
            ],
            [[-0.75, 0.0], [-0.25, 0.0], [0.25, 0.0], [0.75, 0.0]],
            [
                [-0.75, 0.666667],
                [-0.25, 0.666667],
                [0.25, 0.666667],
                [0.75, 0.666667],
            ],
        ],
        [
            [
                [-0.108333, -0.958333],
                [0.141667, -0.808333],
                [0.391667, -0.658333],
                [0.641667, -0.508333],
            ],
            [
                [-0.275, -0.425],
                [-0.025, -0.275],
                [0.225, -0.125],
                [0.475, 0.025],
            ],
            [
                [-0.441667, 0.108333],
                [-0.191667, 0.258333],
                [0.058333, 0.408333],
                [0.308333, 0.558333],
            ],
        ],
    ];

    const GRID_ALIGN_CORNERS: [[[[f32; 2]; 4]; 3]; 2] = [
        [
            [
                [-1.0, -1.0],
                [-0.333333, -1.0],
                [0.333333, -1.0],
                [1.0, -1.0],
            ],
            [[-1.0, 0.0], [-0.333333, 0.0], [0.333333, 0.0], [1.0, 0.0]],
            [[-1.0, 1.0], [-0.333333, 1.0], [0.333333, 1.0], [1.0, 1.0]],
        ],
        [
            [
                [-0.15, -1.3],
                [0.183333, -1.1],
                [0.516667, -0.9],
                [0.85, -0.7],
            ],
            [
                [-0.4, -0.5],
                [-0.066667, -0.3],
                [0.266667, -0.1],
                [0.6, 0.1],
            ],
            [[-0.65, 0.3], [-0.316667, 0.5], [0.016667, 0.7], [0.35, 0.9]],
        ],
    ];

    const GRID_SINGLE_ROW: [[[[f32; 2]; 3]; 1]; 2] = [
        [[[-0.666667, 0.0], [0.0, 0.0], [0.666667, 0.0]]],
        [[[-0.233333, -0.4], [0.1, -0.2], [0.433333, 0.0]]],
    ];

    const GRID_SINGLE_ROW_ALIGN_CORNERS: [[[[f32; 2]; 3]; 1]; 2] = [
        [[[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]]],
        [[[-0.15, -1.3], [0.35, -1.0], [0.85, -0.7]]],
    ];

    const GRID_3D: [[[[[f32; 3]; 2]; 3]; 2]; 1] = [[
        [
            [
                [-0.316667, -0.833333, -0.883333],
                [0.583333, -0.933333, -0.883333],
            ],
            [[-0.25, -0.1, -0.75], [0.65, -0.2, -0.75]],
            [
                [-0.183333, 0.633333, -0.616667],
                [0.716667, 0.533333, -0.616667],
            ],
        ],
        [
            [
                [-0.316667, -0.533333, -0.183333],
                [0.583333, -0.633333, -0.183333],
            ],
            [[-0.25, 0.2, -0.05], [0.65, 0.1, -0.05]],
            [
                [-0.183333, 0.933333, 0.083333],
                [0.716667, 0.833333, 0.083333],
            ],
        ],
    ]];

    const GRID_3D_ALIGN_CORNERS: [[[[[f32; 3]; 2]; 3]; 2]; 1] = [[
        [
            [[-0.8, -1.3, -1.3], [1.0, -1.5, -1.3]],
            [[-0.7, -0.2, -1.1], [1.1, -0.4, -1.1]],
            [[-0.6, 0.9, -0.9], [1.2, 0.7, -0.9]],
        ],
        [
            [[-0.8, -0.7, 0.1], [1.0, -0.9, 0.1]],
            [[-0.7, 0.4, 0.3], [1.1, 0.2, 0.3]],
            [[-0.6, 1.5, 0.5], [1.2, 1.3, 0.5]],
        ],
    ]];
}
