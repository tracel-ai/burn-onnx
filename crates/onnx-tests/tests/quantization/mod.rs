use crate::include_models;
include_models!(
    scalar,
    dynamic,
    quant_conv1d,
    quant_conv2d,
    quant_conv3d,
    large_accumulation
);
#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};
    #[test]
    #[cfg_attr(feature = "test-wgpu", ignore = "WGSL has no 8-bit integers")]
    fn scalar_scale_and_zero_point() {
        let device = Default::default();
        let model =
            scalar::Model::from_file(concat!(env!("OUT_DIR"), "/model/scalar.bpk"), &device);
        let x = Tensor::<2>::from_floats([[-100.0, -1.25, -0.25], [0.25, 1.25, 1000.0]], &device);
        let (q, y) = model.forward(x, 0.5, 4);
        q.into_data()
            .assert_eq(&TensorData::from([[0u8, 2, 4], [4, 6, 255]]), true);
        y.into_data()
            .assert_eq(&TensorData::from([[-2f32, -1., 0.], [0., 1., 125.5]]), true);
    }
    #[test]
    #[cfg_attr(feature = "test-wgpu", ignore = "WGSL has no 8-bit integers")]
    fn dynamic_quantization() {
        let device = Default::default();
        let model =
            dynamic::Model::from_file(concat!(env!("OUT_DIR"), "/model/dynamic.bpk"), &device);
        let (y, scale, zero) = model.forward(Tensor::<2>::from_floats(
            [[-2., -1., 0.], [1., 2., 3.]],
            &device,
        ));
        y.into_data()
            .assert_eq(&TensorData::from([[0u8, 51, 102], [153, 204, 255]]), true);
        assert!((scale - 5.0 / 255.0).abs() < 1e-7);
        assert_eq!(zero, 102);
        let (y, scale, zero) = model.forward(Tensor::<2>::from_floats([[0.; 3]; 2], &device));
        y.into_data()
            .assert_eq(&TensorData::from([[0u8; 3]; 2]), true);
        assert_eq!(scale, 1.0 / 255.0);
        assert_eq!(zero, 0);
    }
    include!("convolution_data.rs");

    #[test]
    #[cfg_attr(feature = "test-wgpu", ignore = "WGSL has no 8-bit integers")]
    fn grouped_dilated_conv1d() {
        let device = Default::default();
        let model = quant_conv1d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/quant_conv1d.bpk"),
            &device,
        );
        let input = Tensor::<3, burn::tensor::Int>::from_data(
            TensorData::new(INPUT_1D.to_vec(), [1, 4, 4]),
            (&device, burn::tensor::DType::U8),
        );
        let (integer, quantized) = model.forward(input);
        integer
            .into_data()
            .assert_eq(&TensorData::new(INTEGER_1D.to_vec(), [1, 4, 2]), true);
        quantized
            .into_data()
            .assert_eq(&TensorData::new(QUANTIZED_1D.to_vec(), [1, 4, 2]), true);
    }

    #[test]
    #[cfg_attr(feature = "test-wgpu", ignore = "WGSL has no 8-bit integers")]
    fn grouped_dilated_conv2d() {
        let device = Default::default();
        let model = quant_conv2d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/quant_conv2d.bpk"),
            &device,
        );
        let input = Tensor::<4, burn::tensor::Int>::from_data(
            TensorData::new(INPUT_2D.to_vec(), [1, 4, 4, 4]),
            (&device, burn::tensor::DType::U8),
        );
        let (integer, quantized) = model.forward(input);
        integer
            .into_data()
            .assert_eq(&TensorData::new(INTEGER_2D.to_vec(), [1, 4, 2, 2]), true);
        quantized
            .into_data()
            .assert_eq(&TensorData::new(QUANTIZED_2D.to_vec(), [1, 4, 2, 2]), true);
    }

    #[test]
    #[cfg_attr(feature = "test-wgpu", ignore = "WGSL has no 8-bit integers")]
    fn grouped_dilated_conv3d() {
        let device = Default::default();
        let model = quant_conv3d::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/quant_conv3d.bpk"),
            &device,
        );
        let input = Tensor::<5, burn::tensor::Int>::from_data(
            TensorData::new(INPUT_3D.to_vec(), [1, 4, 4, 4, 4]),
            (&device, burn::tensor::DType::U8),
        );
        let (integer, quantized) = model.forward(input);
        integer
            .into_data()
            .assert_eq(&TensorData::new(INTEGER_3D.to_vec(), [1, 4, 2, 2, 2]), true);
        quantized.into_data().assert_eq(
            &TensorData::new(QUANTIZED_3D.to_vec(), [1, 4, 2, 2, 2]),
            true,
        );
    }
    #[test]
    #[cfg_attr(feature = "test-wgpu", ignore = "WGSL has no 8-bit integers")]
    fn accumulation_preserves_int32_precision() {
        let device = Default::default();
        let model = large_accumulation::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/large_accumulation.bpk"),
            &device,
        );
        let input = Tensor::<3, burn::tensor::Int>::full(
            [1, 512, 1],
            255,
            (&device, burn::tensor::DType::U8),
        );
        model
            .forward(input)
            .into_data()
            .assert_eq(&TensorData::from([[[33292545i32]]]), true);
    }

    #[test]
    #[cfg_attr(feature = "test-wgpu", ignore = "WGSL has no 8-bit integers")]
    fn dynamic_quantization_includes_zero_in_range() {
        let device = Default::default();
        let model =
            dynamic::Model::from_file(concat!(env!("OUT_DIR"), "/model/dynamic.bpk"), &device);
        for (input, expected, expected_zero) in [
            (
                [[1., 2., 3.], [4., 5., 6.]],
                [[42u8, 85, 128], [170, 212, 255]],
                0u8,
            ),
            (
                [[-6., -5., -4.], [-3., -2., -1.]],
                [[0u8, 43, 85], [127, 170, 213]],
                255u8,
            ),
        ] {
            let (y, scale, zero) = model.forward(Tensor::<2>::from_floats(input, &device));
            y.into_data().assert_eq(&TensorData::from(expected), true);
            assert_eq!(scale, 6.0 / 255.0);
            assert_eq!(zero, expected_zero);
        }
    }
}
