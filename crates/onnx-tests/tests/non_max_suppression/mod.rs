use crate::include_models;
include_models!(
    non_max_suppression,
    non_max_suppression_center,
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

    fn corner_boxes(device: &Device) -> Tensor<3> {
        Tensor::from_floats(
            [[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]],
            device,
        )
    }

    fn center_boxes(device: &Device) -> Tensor<3> {
        Tensor::from_floats(
            [[
                [0.5, 0.5, 1.0, 1.0],
                [0.6, 0.5, 1.0, 1.0],
                [0.4, 0.5, 1.0, 1.0],
                [10.5, 0.5, 1.0, 1.0],
                [10.6, 0.5, 1.0, 1.0],
                [100.5, 0.5, 1.0, 1.0],
            ]],
            device,
        )
    }

    fn scores(device: &Device) -> Tensor<3> {
        Tensor::from_floats([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], device)
    }

    fn int_scalar(value: i64, device: &Device) -> Tensor<1, Int> {
        Tensor::from_data(TensorData::from([value]), (device, DType::I64))
    }

    #[test]
    fn corner_format() {
        let device = Device::default();
        let model = load_model!(non_max_suppression, &device);
        let output = model.forward(
            corner_boxes(&device),
            scores(&device),
            int_scalar(3, &device),
            Tensor::from_floats([0.5], &device),
            Tensor::from_floats([0.0], &device),
        );

        output.to_data().assert_eq(
            &TensorData::from([[0i64, 0, 3], [0, 0, 0], [0, 0, 5]]),
            true,
        );
    }

    #[test]
    fn center_format() {
        let device = Device::default();
        let model = load_model!(non_max_suppression_center, &device);
        let output = model.forward(
            center_boxes(&device),
            scores(&device),
            int_scalar(3, &device),
            Tensor::from_floats([0.5], &device),
            Tensor::from_floats([0.0], &device),
        );

        output.to_data().assert_eq(
            &TensorData::from([[0i64, 0, 3], [0, 0, 0], [0, 0, 5]]),
            true,
        );
    }

    #[test]
    fn omitted_middle_input_uses_default_iou_threshold() {
        let device = Device::default();
        let model = load_model!(non_max_suppression_missing_middle, &device);
        let output = model.forward(
            corner_boxes(&device),
            scores(&device),
            int_scalar(3, &device),
            Tensor::from_floats([0.8], &device),
        );

        output
            .to_data()
            .assert_eq(&TensorData::from([[0i64, 0, 3], [0, 0, 0]]), true);
    }

    #[test]
    fn omitted_score_threshold_keeps_negative_scores() {
        let device = Device::default();
        let model = load_model!(non_max_suppression_missing_score_threshold, &device);
        let boxes = corner_boxes(&device).slice([0..1, 0..2, 0..4]);
        let output = model.forward(
            boxes,
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
        let boxes = Tensor::from_floats(
            [[
                [0.0, 0.0, 1.0, 1.0],
                [10.0, 10.0, 11.0, 11.0],
                [20.0, 20.0, 21.0, 21.0],
                [30.0, 30.0, 31.0, 31.0],
                [40.0, 40.0, 41.0, 41.0],
                [50.0, 50.0, 51.0, 51.0],
            ]],
            &device,
        );
        let scores = Tensor::from_floats([[[0.5, 0.6, 0.4, 0.3, 0.2, 0.1]]], &device);
        let output = model.forward(
            boxes,
            scores,
            int_scalar(3, &device),
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
        let boxes = Tensor::from_floats(
            [[
                [0.0, 0.0, 1.0, 1.0],
                [0.5, 0.5, 1.5, 1.5],
                [10.0, 10.0, 11.0, 11.0],
                [20.0, 20.0, 21.0, 21.0],
                [30.0, 30.0, 31.0, 31.0],
                [40.0, 40.0, 41.0, 41.0],
            ]],
            &device,
        );
        let scores = Tensor::from_floats([[[0.9, 0.8, -0.1, -0.2, -0.3, -0.4]]], &device);
        let exact_iou = 0.25f32 / 1.75f32;
        let output = model.forward(
            boxes,
            scores,
            int_scalar(3, &device),
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
        let boxes = corner_boxes(&device).select(1, int_scalar_indices(&[0, 1, 3, 4], &device));
        let scores = Tensor::from_floats([[[0.9, 0.8, 0.7, 0.6], [0.5, 0.6, 0.9, 0.8]]], &device);
        let output = model.forward(
            boxes,
            scores,
            int_scalar(2, &device),
            Tensor::from_floats([0.5], &device),
            Tensor::from_floats([0.0], &device),
        );

        output.to_data().assert_eq(
            &TensorData::from([[0i64, 0, 0], [0, 0, 2], [0, 1, 2], [0, 1, 1]]),
            true,
        );
    }

    fn int_scalar_indices(values: &[i64], device: &Device) -> Tensor<1, Int> {
        Tensor::from_data(
            TensorData::new(values.to_vec(), [values.len()]),
            (device, DType::I64),
        )
    }

    #[test]
    fn omitted_max_output_returns_empty() {
        let device = Device::default();
        let model = load_model!(non_max_suppression_minimal, &device);
        let boxes = corner_boxes(&device).slice([0..1, 0..2, 0..4]);
        let scores = Tensor::from_floats([[[0.9, 0.8]]], &device);
        let output = model.forward(boxes, scores);

        assert_eq!(output.shape().dims(), [0, 3]);
        assert_eq!(output.dtype(), DType::I64);
    }

    #[test]
    fn reversed_corner_coordinates_are_normalized() {
        let device = Device::default();
        let model = load_model!(non_max_suppression, &device);
        let boxes = Tensor::from_floats(
            [[
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.1, 1.0, 1.1],
                [10.0, 10.0, 11.0, 11.0],
                [20.0, 20.0, 21.0, 21.0],
                [30.0, 30.0, 31.0, 31.0],
                [40.0, 40.0, 41.0, 41.0],
            ]],
            &device,
        );
        let scores = Tensor::from_floats([[[0.9, 0.8, 0.0, 0.0, 0.0, 0.0]]], &device);
        let output = model.forward(
            boxes,
            scores,
            int_scalar(2, &device),
            Tensor::from_floats([0.5], &device),
            Tensor::from_floats([0.1], &device),
        );

        output
            .to_data()
            .assert_eq(&TensorData::from([[0i64, 0, 0]]), true);
    }
}
