// Import the shared macro
use crate::include_models;
include_models!(
    maxpool1d,
    maxpool1d_asymmetric_padding,
    maxpool1d_indices,
    maxpool2d_indices,
    maxpool2d_indices_same,
    maxpool2d_indices_ceil,
    maxpool1d_ceil_mode,
    maxpool2d,
    maxpool2d_asymmetric_padding,
    maxpool2d_same_upper_dynamic,
    maxpool2d_ceil_mode
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, Int, Shape, Tensor, TensorData};

    #[test]
    fn maxpool1d() {
        let device = Default::default();

        let model: maxpool1d::Model = maxpool1d::Model::new(&device);
        let input = Tensor::<3>::from_floats(
            [[
                [1.927, 1.487, 0.901, -2.106, 0.678],
                [-1.235, -0.043, -1.605, -0.752, -0.687],
                [-0.493, 0.241, -1.111, 0.092, -2.317],
                [-0.217, -1.385, -0.396, 0.803, -0.622],
                [-0.592, -0.063, -0.829, 0.331, -1.558],
            ]],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[
            [1.927f32, 1.927, 0.901],
            [-0.043, -0.043, -0.687],
            [0.241, 0.241, 0.092],
            [-0.217, 0.803, 0.803],
            [-0.063, 0.331, 0.331],
        ]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool2d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: maxpool2d::Model = maxpool2d::Model::new(&device);

        // Run the model
        let input = Tensor::<4>::from_floats(
            [[[
                [1.927, 1.487, 0.901, -2.106, 0.678],
                [-1.235, -0.043, -1.605, -0.752, -0.687],
                [-0.493, 0.241, -1.111, 0.092, -2.317],
                [-0.217, -1.385, -0.396, 0.803, -0.622],
                [-0.592, -0.063, -0.829, 0.331, -1.558],
            ]]],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[[
            [0.901f32, 1.927, 1.487, 0.901],
            [0.901, 1.927, 1.487, 0.901],
            [-0.396, 0.803, 0.241, -0.396],
        ]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool1d_ceil_mode() {
        // Test ceil_mode=True for MaxPool1d
        // Input: 1x1x6 (values 1-6), kernel: 3, stride: 2, padding: 0
        // With ceil_mode=True: output = ceil((6-3)/2)+1 = 3 elements
        let device = Default::default();
        let model: maxpool1d_ceil_mode::Model = maxpool1d_ceil_mode::Model::new(&device);

        let input = Tensor::<3>::from_floats([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], &device);
        let output = model.forward(input);

        // Window 0: max(1,2,3) = 3
        // Window 1: max(3,4,5) = 5
        // Window 2: max(5,6) = 6 (partial window at edge)
        let expected = TensorData::from([[[3.0f32, 5.0, 6.0]]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool2d_ceil_mode() {
        // Test ceil_mode=True for MaxPool2d
        // Input: 1x1x6x6 (values 1-36), kernel: 3x3, stride: 2x2, padding: 0
        // With ceil_mode=True: output = 3x3
        let device = Default::default();
        let model: maxpool2d_ceil_mode::Model = maxpool2d_ceil_mode::Model::new(&device);

        let input = Tensor::<4>::from_floats(
            [[[
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
            ]]],
            &device,
        );
        let output = model.forward(input);

        // With ceil_mode=True, we get 3x3 output instead of 2x2
        // (0,0): max of rows 0-2, cols 0-2 = 15
        // (0,1): max of rows 0-2, cols 2-4 = 17
        // (0,2): max of rows 0-2, cols 4-5 = 18
        // (1,0): max of rows 2-4, cols 0-2 = 27
        // (1,1): max of rows 2-4, cols 2-4 = 29
        // (1,2): max of rows 2-4, cols 4-5 = 30
        // (2,0): max of rows 4-5, cols 0-2 = 33
        // (2,1): max of rows 4-5, cols 2-4 = 35
        // (2,2): max of rows 4-5, cols 4-5 = 36
        let expected = TensorData::from([[[
            [15.0f32, 17.0, 18.0],
            [27.0, 29.0, 30.0],
            [33.0, 35.0, 36.0],
        ]]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool1d_asymmetric_padding() {
        // Test asymmetric padding (left=1, right=2) for MaxPool1d
        let device = Default::default();
        let model: maxpool1d_asymmetric_padding::Model =
            maxpool1d_asymmetric_padding::Model::new(&device);

        // Run the model with ones as input for easier testing
        let input = Tensor::<3>::ones([2, 4, 10], &device);
        let output = model.forward(input);

        // With asymmetric padding (1, 2), input length 10 becomes 10+1+2=13
        // After pool with kernel 3, stride 1, output length is 13-3+1=11
        let expected_shape = Shape::from([2, 4, 11]);
        assert_eq!(output.shape(), expected_shape);

        // Verify the sum matches PyTorch output (all 1.0 values, so max = 1.0 for all positions that see valid input)
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum: f32 = 88.0; // from pytorch
        assert!(
            (output_sum - expected_sum).abs() < 0.1,
            "Expected sum ~{}, got {}",
            expected_sum,
            output_sum
        );
    }

    #[test]
    fn maxpool2d_asymmetric_padding() {
        // Test asymmetric padding: top=1, left=1, bottom=2, right=2 (ONNX pads: [1,1,2,2])
        let device = Default::default();
        let model: maxpool2d_asymmetric_padding::Model =
            maxpool2d_asymmetric_padding::Model::new(&device);

        // Run the model with ones as input for easier testing
        let input = Tensor::<4>::ones([2, 4, 10, 15], &device);
        let output = model.forward(input);

        // With asymmetric padding (1, 1, 2, 2), input (10, 15) becomes (13, 18)
        // After pool with kernel (3, 3), stride (1, 1), output is (11, 16)
        let expected_shape = Shape::from([2, 4, 11, 16]);
        assert_eq!(output.shape(), expected_shape);

        // Verify the sum matches ONNX Runtime output
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum: f32 = 1408.0; // from ONNX Runtime
        assert!(
            (output_sum - expected_sum).abs() < 1.0,
            "Expected sum ~{}, got {}",
            expected_sum,
            output_sum
        );
    }

    #[test]
    fn maxpool2d_same_upper_dynamic() {
        // auto_pad=SAME_UPPER over dynamic H/W: kernel 2, stride 2 on an extent of 5 leaves a
        // total padding of 1, which SAME_UPPER puts at the end, so the windows are
        // [0,1] [2,3] [4,pad]. SAME_LOWER would pad the start and shift every window:
        // [pad,0] [1,2] [3,4].
        let device = Default::default();
        let model: maxpool2d_same_upper_dynamic::Model =
            maxpool2d_same_upper_dynamic::Model::from_file(
                concat!(env!("OUT_DIR"), "/model/maxpool2d_same_upper_dynamic.bpk"),
                &device,
            );

        let input = Tensor::<1, Int>::arange(1..51, &device)
            .reshape([1, 2, 5, 5])
            .float();
        let output = model.forward(input);

        // Ground truth from onnx.reference.ReferenceEvaluator
        // (maxpool2d_same_upper_dynamic.py).
        let expected = TensorData::from([[
            [[7.0f32, 9.0, 10.0], [17.0, 19.0, 20.0], [22.0, 24.0, 25.0]],
            [[32.0, 34.0, 35.0], [42.0, 44.0, 45.0], [47.0, 49.0, 50.0]],
        ]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool1d_indices() {
        // Symmetric pads (column-major storage_order, the same as row-major in 1D),
        // asymmetric pads, ceil_mode dropping a window that starts in the trailing
        // padding, ceil_mode keeping a partial window (asymmetric pads, dilation 2),
        // SAME_UPPER on an input whose length is only known at run time, and uint8 /
        // int8 inputs. Expected values and indices from maxpool1d_indices.py (checked
        // against onnxruntime).
        let device = Default::default();
        let model = maxpool1d_indices::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/maxpool1d_indices.bpk"),
            &device,
        );
        let data = [
            [[9, 25, 8, 21, 0, 12, 17], [22, 11, 13, 15, 1, 4, 5]],
            [[2, 16, 23, 3, 26, 24, 18], [27, 20, 7, 10, 14, 19, 6]],
        ];
        let x = || {
            Tensor::<3>::from_data(
                TensorData::from(data.map(|b| b.map(|c| c.map(|v| v as f32)))),
                &device,
            )
        };
        let xu = Tensor::<3, Int>::from_data(
            TensorData::from(data.map(|b| b.map(|c| c.map(|v| v as u8)))),
            (&device, DType::U8),
        );
        let xi = Tensor::<3, Int>::from_data(
            TensorData::from(data.map(|b| b.map(|c| c.map(|v| (v - 14) as i8)))),
            (&device, DType::I8),
        );

        let (
            y_sym,
            i_sym,
            y_asym,
            i_asym,
            y_ceil,
            i_ceil,
            y_dil,
            i_dil,
            y_same,
            i_same,
            y_uint8,
            i_uint8,
            y_int8,
        ) = model.forward(x(), x(), xu, xi);

        let expected_sym = TensorData::from([
            [[9.0f32, 25., 21., 17.], [22., 13., 15., 5.]],
            [[2., 23., 26., 24.], [27., 20., 14., 19.]],
        ]);
        let expected_sym_indices = TensorData::from([
            [[0i64, 1, 3, 6], [7, 9, 10, 13]],
            [[14, 16, 18, 19], [21, 22, 25, 26]],
        ]);
        y_sym.to_data().assert_eq(&expected_sym, true);
        i_sym.to_data().assert_eq(&expected_sym_indices, true);
        // Same kernel, stride and pads as sym: ceil_mode adds a fifth window that
        // would start in the trailing padding, which ONNX drops, leaving sym's four.
        y_ceil.to_data().assert_eq(&expected_sym, true);
        i_ceil.to_data().assert_eq(&expected_sym_indices, true);
        // The sym configuration on uint8 keeps the input dtype.
        y_uint8.to_data().assert_eq(
            &TensorData::from([
                [[9u8, 25, 21, 17], [22, 13, 15, 5]],
                [[2, 23, 26, 24], [27, 20, 14, 19]],
            ]),
            true,
        );
        i_uint8.to_data().assert_eq(&expected_sym_indices, true);

        y_asym.to_data().assert_eq(
            &TensorData::from([
                [
                    [25.0f32, 25., 21., 21., 17., 17., 17.],
                    [22., 15., 15., 15., 5., 5., 5.],
                ],
                [
                    [23., 23., 26., 26., 26., 24., 18.],
                    [27., 20., 14., 19., 19., 19., 6.],
                ],
            ]),
            true,
        );
        i_asym.to_data().assert_eq(
            &TensorData::from([
                [[1i64, 1, 3, 3, 6, 6, 6], [7, 10, 10, 10, 13, 13, 13]],
                [[16, 16, 18, 18, 18, 19, 20], [21, 22, 25, 26, 26, 26, 27]],
            ]),
            true,
        );

        // The last window covers position 6 and the right pad.
        y_dil.to_data().assert_eq(
            &TensorData::from([
                [[9.0f32, 8., 17., 17.], [22., 13., 5., 5.]],
                [[23., 26., 26., 18.], [27., 14., 14., 6.]],
            ]),
            true,
        );
        i_dil.to_data().assert_eq(
            &TensorData::from([
                [[0i64, 2, 6, 6], [7, 9, 13, 13]],
                [[16, 18, 18, 20], [21, 25, 25, 27]],
            ]),
            true,
        );

        y_same.to_data().assert_eq(
            &TensorData::from([
                [
                    [25.0f32, 25., 21., 21., 12., 17., 17.],
                    [22., 13., 15., 15., 4., 5., 5.],
                ],
                [
                    [16., 23., 23., 26., 26., 24., 18.],
                    [27., 20., 10., 14., 19., 19., 6.],
                ],
            ]),
            true,
        );
        i_same.to_data().assert_eq(
            &TensorData::from([
                [[1i64, 1, 3, 3, 5, 6, 6], [7, 9, 10, 10, 12, 13, 13]],
                [[15, 16, 16, 18, 18, 19, 20], [21, 22, 24, 25, 26, 26, 27]],
            ]),
            true,
        );

        y_int8.to_data().assert_eq(
            &TensorData::from([
                [[11i8, 11, 7, 7, 3], [8, 1, 1, 1, -9]],
                [[9, 9, 12, 12, 12], [13, 6, 0, 5, 5]],
            ]),
            true,
        );
    }

    #[test]
    fn maxpool2d_indices() {
        let device = Default::default();
        let model: maxpool2d_indices::Model = maxpool2d_indices::Model::new(&device);
        let input = Tensor::<4>::from_data(
            TensorData::from([
                [
                    [
                        [30.0f32, 0., 22., 31., 18.],
                        [28., 10., 70., 4., 12.],
                        [49., 33., 67., 35., 68.],
                        [45., 73., 61., 55., 40.],
                    ],
                    [
                        [9., 64., 5., 47., 34.],
                        [62., 42., 54., 16., 39.],
                        [56., 79., 7., 50., 53.],
                        [19., 66., 25., 44., 13.],
                    ],
                ],
                [
                    [
                        [76., 3., 17., 38., 8.],
                        [65., 6., 36., 72., 58.],
                        [46., 78., 15., 27., 41.],
                        [26., 48., 24., 43., 77.],
                    ],
                    [
                        [57., 11., 32., 75., 59.],
                        [63., 69., 37., 29., 1.],
                        [52., 21., 2., 23., 74.],
                        [20., 60., 71., 14., 51.],
                    ],
                ],
            ]),
            &device,
        );

        let (values, indices, values_col, indices_col) = model.forward(input);

        let expected_values = TensorData::from([
            [
                [[30.0f32, 22., 31.], [49., 70., 68.], [45., 73., 55.]],
                [[9., 64., 47.], [62., 79., 53.], [19., 66., 44.]],
            ],
            [
                [[76., 17., 38.], [65., 78., 72.], [26., 48., 77.]],
                [[57., 32., 75.], [63., 69., 74.], [20., 71., 51.]],
            ],
        ]);
        values.to_data().assert_eq(&expected_values, true);
        values_col.to_data().assert_eq(&expected_values, true);
        indices.to_data().assert_eq(
            &TensorData::from([
                [
                    [[0i64, 2, 3], [10, 7, 14], [15, 16, 18]],
                    [[20, 21, 23], [25, 31, 34], [35, 36, 38]],
                ],
                [
                    [[40, 42, 43], [45, 51, 48], [55, 56, 59]],
                    [[60, 62, 63], [65, 66, 74], [75, 77, 79]],
                ],
            ]),
            true,
        );
        indices_col.to_data().assert_eq(
            &TensorData::from([
                [
                    [[0i64, 8, 12], [2, 9, 18], [3, 7, 15]],
                    [[20, 24, 32], [21, 26, 38], [23, 27, 35]],
                ],
                [
                    [[40, 48, 52], [41, 46, 53], [43, 47, 59]],
                    [[60, 68, 72], [61, 65, 78], [63, 71, 79]],
                ],
            ]),
            true,
        );
    }

    #[test]
    fn maxpool2d_indices_same_padding() {
        // SAME_UPPER with a 2x2 kernel pads unevenly; the dynamic input takes the
        // run-time padding path for both MaxPool and the runtime-weight Conv.
        let device = Default::default();
        let model: maxpool2d_indices_same::Model = maxpool2d_indices_same::Model::new(&device);
        let x = || {
            Tensor::<4>::from_data(
                TensorData::from([[
                    [
                        [8.0f32, 16., 0., 18.],
                        [11., 9., 13., 1.],
                        [21., 5., 2., 12.],
                    ],
                    [[15., 3., 4., 22.], [17., 20., 23., 7.], [10., 14., 19., 6.]],
                ]]),
                &device,
            )
        };
        let w = Tensor::<1, Int>::arange(0..16, &device)
            .float()
            .reshape([2, 2, 2, 2])
            .mul_scalar(0.1)
            .sub_scalar(0.7);

        let (values, indices, dynamic_values, dynamic_indices, conv) = model.forward(x(), x(), w);

        let expected_values = TensorData::from([[
            [
                [16.0f32, 16., 18., 18.],
                [21., 13., 13., 12.],
                [21., 5., 12., 12.],
            ],
            [
                [20., 23., 23., 22.],
                [20., 23., 23., 7.],
                [14., 19., 19., 6.],
            ],
        ]]);
        values.to_data().assert_eq(&expected_values, true);
        dynamic_values.to_data().assert_eq(&expected_values, true);
        indices.to_data().assert_eq(
            &TensorData::from([[
                [[1i64, 1, 3, 3], [8, 6, 6, 11], [8, 9, 11, 11]],
                [[17, 18, 18, 15], [17, 18, 18, 19], [21, 22, 22, 23]],
            ]]),
            true,
        );
        dynamic_indices.to_data().assert_eq(
            &TensorData::from([[
                [[3i64, 3, 9, 9], [2, 7, 7, 11], [2, 5, 11, 11]],
                [[16, 19, 19, 21], [16, 19, 19, 22], [17, 20, 20, 23]],
            ]]),
            true,
        );
        conv.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [
                    [-31.1f32, -24.6, -25.6, -20.4],
                    [-35.7, -29.4, -25.7, -9.4],
                    [-23.5, -12.7, -15.5, -10.2],
                ],
                [
                    [48.1, 45.8, 44.8, 18.0],
                    [49.9, 54.6, 40.7, 11.4],
                    [16.5, 19.3, 15.7, 4.2],
                ],
            ]]),
            burn::tensor::Tolerance::absolute(1e-3),
        );
    }

    #[test]
    fn maxpool2d_indices_ceil_mode_drops_padding_window() {
        // ONNX drops a ceil-mode window that would start in the trailing padding.
        // Expected values from maxpool2d_indices_ceil.py (checked against the onnx
        // reference), indices from the definition.
        let device = Default::default();
        let model: maxpool2d_indices_ceil::Model = maxpool2d_indices_ceil::Model::new(&device);
        // Rows of 0..25 in reverse row order.
        let values: alloc::vec::Vec<f32> = (0..5)
            .rev()
            .flat_map(|row| (0..5).map(move |col| (row * 5 + col) as f32))
            .collect();
        let x = Tensor::<4>::from_data(TensorData::new(values, [1, 1, 5, 5]), &device);

        let (y_sym, i_sym, y_asym, i_asym) = model.forward(x);

        y_sym.to_data().assert_eq(
            &TensorData::from([[[[20.0f32, 22., 24.], [15., 17., 19.], [5., 7., 9.]]]]),
            true,
        );
        i_sym.to_data().assert_eq(
            &TensorData::from([[[[0i64, 2, 4], [5, 7, 9], [15, 17, 19]]]]),
            true,
        );
        y_asym
            .to_data()
            .assert_eq(&TensorData::from([[[[22.0f32, 24.], [7., 9.]]]]), true);
        i_asym
            .to_data()
            .assert_eq(&TensorData::from([[[[2i64, 4], [17, 19]]]]), true);
    }
}
