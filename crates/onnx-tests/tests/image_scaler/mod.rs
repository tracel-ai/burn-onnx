use crate::include_models;
include_models!(
    image_scaler,
    image_scaler_default,
    image_scaler_f64,
    image_scaler_nonfinite
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn scales_and_biases_each_channel() {
        let device = Default::default();
        let input = Tensor::<4>::from_data(
            TensorData::new(
                (-12..12).map(|x| x as f32).collect::<alloc::vec::Vec<_>>(),
                [2, 3, 2, 2],
            ),
            &device,
        );
        let model = image_scaler::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/image_scaler.bpk"),
            &device,
        );
        let expected = TensorData::new(
            alloc::vec![
                -5.0f32, -4.5, -4., -3.5, -6., -5.5, -5.0f32, -4.5, 1., 1.5, 2., 2.5, 1., 1.5, 2.,
                2.5, 0., 0.5, 1., 1.5, 7., 7.5, 8., 8.5
            ],
            [2, 3, 2, 2],
        );
        assert_eq!(model.forward(input.clone()).into_data(), expected);
        let identity = image_scaler_default::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/image_scaler_default.bpk"),
            &device,
        );
        assert_eq!(
            identity.forward(input.clone()).into_data(),
            input.into_data()
        );
    }
}

#[test]
fn attributes_preserve_f32_values_in_f64_models() {
    use burn::tensor::{DType, Tensor, TensorData};
    let device = burn::tensor::Device::default();
    if !device.supports_dtype(DType::F64) {
        return;
    }
    let model = image_scaler_f64::Model::from_file(
        concat!(env!("OUT_DIR"), "/model/image_scaler_f64.bpk"),
        &device,
    );
    let input = Tensor::<4>::from_data(
        TensorData::new(alloc::vec![0f64], [1, 1, 1, 1]),
        (&device, DType::F64),
    );
    assert_eq!(
        model.forward(input).into_data().to_vec::<f64>().unwrap(),
        [0.1f32 as f64]
    );
    let model = image_scaler_nonfinite::Model::from_file(
        concat!(env!("OUT_DIR"), "/model/image_scaler_nonfinite.bpk"),
        &device,
    );
    let input = Tensor::<4>::from_data(
        TensorData::new(alloc::vec![0f64; 2], [1, 2, 1, 1]),
        (&device, DType::F64),
    );
    assert!(
        model
            .forward(input)
            .into_data()
            .to_vec::<f64>()
            .unwrap()
            .iter()
            .all(|v| v.is_nan())
    );
}
