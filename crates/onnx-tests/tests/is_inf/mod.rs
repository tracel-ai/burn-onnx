// Import the shared macro
use crate::include_models;
include_models!(
    is_inf,
    is_inf_scalar,
    is_inf_neg_only,
    is_inf_pos_only,
    is_inf_none
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    #[cfg_attr(feature = "test-metal", ignore = "Metal has no f64")]
    fn is_inf() {
        let device = Device::default();
        // The model casts to f64; support on wgpu depends on the adapter.
        if !device.supports_dtype(burn::tensor::DType::F64) {
            return;
        }
        let model: is_inf::Model = is_inf::Model::new(&device);

        let input1 =
            Tensor::<2>::from_floats([[1.0, f32::INFINITY, -9.0, f32::NEG_INFINITY]], &device);

        let output = model.forward(input1);
        let expected = TensorData::from([[false, true, false, true]]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn is_inf_scalar() {
        let device = Default::default();
        let model: is_inf_scalar::Model = is_inf_scalar::Model::new(&device);

        let input1 = f32::INFINITY;

        let output = model.forward(input1);
        let expected = true;

        assert_eq!(output, expected);
    }

    #[test]
    fn is_inf_neg_only() {
        let device = Default::default();
        let model: is_inf_neg_only::Model = is_inf_neg_only::Model::new(&device);

        let input1 =
            Tensor::<2>::from_floats([[1.0, f32::INFINITY, -9.0, f32::NEG_INFINITY]], &device);

        let output = model.forward(input1);
        let expected = TensorData::from([[false, false, false, true]]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn is_inf_pos_only() {
        let device = Default::default();
        let model: is_inf_pos_only::Model = is_inf_pos_only::Model::new(&device);

        let input1 =
            Tensor::<2>::from_floats([[1.0, f32::INFINITY, -9.0, f32::NEG_INFINITY]], &device);

        let output = model.forward(input1);
        let expected = TensorData::from([[false, true, false, false]]);

        output.to_data().assert_eq(&expected, false);
    }

    #[test]
    fn is_inf_none() {
        let device = Default::default();
        let model: is_inf_none::Model = is_inf_none::Model::new(&device);

        let input1 =
            Tensor::<2>::from_floats([[1.0, f32::INFINITY, -9.0, f32::NEG_INFINITY]], &device);

        let output = model.forward(input1);
        let expected = TensorData::from([[false, false, false, false]]);

        output.to_data().assert_eq(&expected, false);
    }
}
