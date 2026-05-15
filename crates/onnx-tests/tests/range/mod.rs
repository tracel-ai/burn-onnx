use crate::include_models;
include_models!(
    range,
    range_static,
    range_mixed,
    range_runtime,
    range_negative_delta
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

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
}
