use crate::include_models;
include_models!(
    non_max_suppression,
    non_max_suppression_minimal,
    non_max_suppression_missing_middle,
    non_max_suppression_missing_score_threshold
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, Int, Tensor, TensorData};

    macro_rules! load_model {
        ($module:ident, $device:expr) => {
            $module::Model::from_file(
                concat!(env!("OUT_DIR"), "/model/", stringify!($module), ".bpk"),
                $device,
            )
        };
    }

    fn overlapping_boxes(device: &Device) -> Tensor<3> {
        Tensor::from_floats([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.5]]], device)
    }

    fn int_scalar(value: i64, device: &Device) -> Tensor<1, Int> {
        Tensor::from_data(TensorData::from([value]), (device, DType::I64))
    }

    #[test]
    fn omitted_middle_input_uses_default_iou_threshold() {
        let device = Device::default();
        let model = load_model!(non_max_suppression_missing_middle, &device);
        let output = model.forward(
            overlapping_boxes(&device),
            Tensor::from_floats([[[0.9, 0.8]]], &device),
            int_scalar(2, &device),
            Tensor::from_floats([0.0], &device),
        );

        output
            .to_data()
            .assert_eq(&TensorData::from([[0i64, 0, 0]]), true);
    }

    #[test]
    fn omitted_score_threshold_keeps_negative_scores() {
        let device = Device::default();
        let model = load_model!(non_max_suppression_missing_score_threshold, &device);
        let output = model.forward(
            overlapping_boxes(&device),
            Tensor::from_floats([[[-0.1, -0.2]]], &device),
            int_scalar(1, &device),
            Tensor::from_floats([0.5], &device),
        );

        output
            .to_data()
            .assert_eq(&TensorData::from([[0i64, 0, 0]]), true);
    }

    #[test]
    fn score_equal_to_threshold_is_removed() {
        let device = Device::default();
        let model = load_model!(non_max_suppression, &device);
        let output = model.forward(
            overlapping_boxes(&device),
            Tensor::from_floats([[[0.5, 0.6]]], &device),
            int_scalar(2, &device),
            Tensor::from_floats([0.0], &device),
            Tensor::from_floats([0.5], &device),
        );

        output
            .to_data()
            .assert_eq(&TensorData::from([[0i64, 0, 1]]), true);
    }

    #[test]
    fn iou_equal_to_threshold_is_kept() {
        let device = Device::default();
        let model = load_model!(non_max_suppression, &device);
        let exact_iou = 0.5f32 / 1.5f32;
        let output = model.forward(
            overlapping_boxes(&device),
            Tensor::from_floats([[[0.9, 0.8]]], &device),
            int_scalar(2, &device),
            Tensor::from_floats([exact_iou], &device),
            Tensor::from_floats([0.0], &device),
        );

        output
            .to_data()
            .assert_eq(&TensorData::from([[0i64, 0, 0], [0, 0, 1]]), true);
    }

    #[test]
    fn multiple_classes() {
        let device = Device::default();
        let model = load_model!(non_max_suppression, &device);
        let output = model.forward(
            overlapping_boxes(&device),
            Tensor::from_floats([[[0.9, 0.8], [0.8, 0.9]]], &device),
            int_scalar(1, &device),
            Tensor::from_floats([0.5], &device),
            Tensor::from_floats([0.0], &device),
        );

        output
            .to_data()
            .assert_eq(&TensorData::from([[0i64, 0, 0], [0, 1, 1]]), true);
    }

    #[test]
    fn omitted_max_output_returns_empty() {
        let device = Device::default();
        let model = load_model!(non_max_suppression_minimal, &device);
        let output = model.forward(
            overlapping_boxes(&device),
            Tensor::from_floats([[[0.9, 0.8]]], &device),
        );

        assert_eq!(output.dims(), [0, 3]);
        assert_eq!(output.dtype(), DType::I64);
    }
}
