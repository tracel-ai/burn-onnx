use crate::include_models;
include_models!(
    range,
    range_static,
    range_mixed,
    range_runtime,
    range_negative_delta,
    range_float_static,
    range_float_mixed,
    range_int32_mixed,
    range_float_runtime,
    range_double_mixed,
    range_int16
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{TensorData, Tolerance};

    #[test]
    fn range() {
        let device = Default::default();
        let model: range::Model = range::Model::new(&device);

        // Run the model
        let start = 0i64;
        let limit = 10i64;
        let delta = 2i64;
        let output = model.forward(start, limit, delta);

        let expected = TensorData::from([0i64, 2, 4, 6, 8]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn range_static() {
        let device = Default::default();
        let model: range_static::Model = range_static::Model::new(&device);

        // Run the model - all parameters are static
        let output = model.forward();

        let expected = TensorData::from([0i64, 2, 4, 6, 8]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn range_mixed() {
        let device = Default::default();
        let model: range_mixed::Model = range_mixed::Model::new(&device);

        // Run the model - start is runtime, limit and delta are static
        let start = 0i64;
        let output = model.forward(start);

        let expected = TensorData::from([0i64, 3, 6, 9, 12]);
        output.to_data().assert_eq(&expected, true);

        // Test with different start value
        let start = 3i64;
        let output = model.forward(start);

        let expected = TensorData::from([3i64, 6, 9, 12]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn range_runtime() {
        let device = Default::default();
        let model: range_runtime::Model = range_runtime::Model::new(&device);

        // Test case 1: 0 to 10 by 2
        let start = 0i64;
        let limit = 10i64;
        let delta = 2i64;
        let output = model.forward(start, limit, delta);

        let expected = TensorData::from([0i64, 2, 4, 6, 8]);
        output.to_data().assert_eq(&expected, true);

        // Test case 2: 5 to 20 by 3
        let start = 5i64;
        let limit = 20i64;
        let delta = 3i64;
        let output = model.forward(start, limit, delta);

        let expected = TensorData::from([5i64, 8, 11, 14, 17]);
        output.to_data().assert_eq(&expected, true);

        // Test case 3: negative delta (descending range 10 to 0 by -2)
        let output = model.forward(10, 0, -2);

        let expected = TensorData::from([10i64, 8, 6, 4, 2]);
        output.to_data().assert_eq(&expected, true);

        // Test case 4: negative delta with non-zero limit
        let output = model.forward(20, 5, -3);

        let expected = TensorData::from([20i64, 17, 14, 11, 8]);
        output.to_data().assert_eq(&expected, true);

        // Test case 5: empty range (start >= limit with positive delta)
        let output = model.forward(10, 0, 2);
        assert_eq!(output.dims(), [0]);

        // Test case 6: empty range (start <= limit with negative delta)
        let output = model.forward(0, 10, -1);
        assert_eq!(output.dims(), [0]);
    }

    #[test]
    fn range_negative_delta() {
        let device = Default::default();
        let model: range_negative_delta::Model = range_negative_delta::Model::new(&device);

        // Descending range: start=10, limit=0, delta=-2
        let output = model.forward();

        let expected = TensorData::from([10i64, 8, 6, 4, 2]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn range_float_static() {
        let device = Default::default();
        let model: range_float_static::Model = range_float_static::Model::new(&device);

        // start=1.5, limit=5.0, delta=0.5: fractional bounds must not be truncated
        let output = model.forward();

        let expected = TensorData::from([1.5f32, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn range_float_mixed() {
        let device = Default::default();
        let model: range_float_mixed::Model = range_float_mixed::Model::new(&device);

        // start=0.5 and delta=0.25 are static, limit is runtime
        let output = model.forward(2.0);
        let expected = TensorData::from([0.5f32, 0.75, 1.0, 1.25, 1.5, 1.75]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());

        let output = model.forward(1.6);
        let expected = TensorData::from([0.5f32, 0.75, 1.0, 1.25, 1.5]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());

        // limit below start: empty
        let output = model.forward(0.2);
        assert_eq!(output.dims(), [0]);
    }

    #[test]
    fn range_int32_mixed() {
        let device = Default::default();
        let model: range_int32_mixed::Model = range_int32_mixed::Model::new(&device);

        // start=1 and delta=3 are static, limit is runtime
        let output = model.forward(11);

        let expected = TensorData::from([1i32, 4, 7, 10]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn range_float_runtime() {
        let device = Default::default();
        let model: range_float_runtime::Model = range_float_runtime::Model::new(&device);

        // limit - start is 2.6 in f32, exactly 2 steps of 1.3, so 2 elements (as in ORT)
        let output = model.forward(-1.5, 1.1, 1.3);
        let expected = TensorData::from([-1.5f32, -0.2]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());

        // Negative fractional delta
        let output = model.forward(2.0, 0.4, -0.5);
        let expected = TensorData::from([2.0f32, 1.5, 1.0, 0.5]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());

        // Empty
        let output = model.forward(1.0, 0.0, 0.5);
        assert_eq!(output.dims(), [0]);
    }

    #[test]
    #[cfg_attr(feature = "test-metal", ignore = "Metal has no f64")]
    fn range_double_mixed() {
        let device = burn::tensor::Device::default();
        // f64 support on wgpu depends on the adapter.
        if !device.supports_dtype(burn::tensor::DType::F64) {
            return;
        }
        let model: range_double_mixed::Model = range_double_mixed::Model::new(&device);

        // start=-0.5 and delta=0.125 are static, limit is runtime
        let output = model.forward(0.0);

        let expected = TensorData::from([-0.5f64, -0.375, -0.25, -0.125]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn range_int16() {
        let device = Default::default();
        let model: range_int16::Model = range_int16::Model::new(&device);

        let output = model.forward();

        let expected = TensorData::from([-3i16, -1, 1]);
        output.to_data().assert_eq(&expected, true);
    }
}
