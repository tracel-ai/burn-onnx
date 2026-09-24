use crate::include_models;
include_models!(pow, pow_int, pow_broadcast, pow_mixed);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, Int, Tensor, TensorData, Tolerance};

    #[test]
    fn pow_int_with_tensor_and_scalar() {
        let device = Default::default();
        let model: pow_int::Model = pow_int::Model::new(&device);

        let input1 = Tensor::<4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let input2 = 2;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[[[1i32, 16, 729, 65536]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn pow_broadcast() {
        let device = Default::default();
        let model: pow_broadcast::Model = pow_broadcast::Model::new(&device);

        // base_3d: all 2.0, shape [2, 3, 4]
        let base_3d = Tensor::from_data(TensorData::from([[[2.0f32; 4]; 3]; 2]), &device);
        // exp_2d: [1, 2, 3, 4] repeated, shape [3, 4]
        let exp_2d = Tensor::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            &device,
        );
        let base_2d = exp_2d.clone();
        let exp_3d = base_3d.clone();

        let (result1, result2) = model.forward(base_3d, exp_2d, base_2d, exp_3d);

        // result1: 2^[1,2,3,4] = [2, 4, 8, 16]
        let expected1 = TensorData::from([
            [
                [2.0f32, 4.0, 8.0, 16.0],
                [2.0, 4.0, 8.0, 16.0],
                [2.0, 4.0, 8.0, 16.0],
            ],
            [
                [2.0, 4.0, 8.0, 16.0],
                [2.0, 4.0, 8.0, 16.0],
                [2.0, 4.0, 8.0, 16.0],
            ],
        ]);

        // result2: [1,2,3,4]^2 = [1, 4, 9, 16]
        let expected2 = TensorData::from([
            [
                [1.0f32, 4.0, 9.0, 16.0],
                [1.0, 4.0, 9.0, 16.0],
                [1.0, 4.0, 9.0, 16.0],
            ],
            [
                [1.0, 4.0, 9.0, 16.0],
                [1.0, 4.0, 9.0, 16.0],
                [1.0, 4.0, 9.0, 16.0],
            ],
        ]);

        result1.to_data().assert_eq(&expected1, true);
        result2.to_data().assert_eq(&expected2, true);
    }

    #[test]
    fn pow_with_tensor_and_scalar() {
        let device = Default::default();
        let model: pow::Model = pow::Model::new(&device);

        let input1 = Tensor::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let input2 = 2f64;

        let output = model.forward(input1, input2);

        let expected = TensorData::from([[[[1.0000f32, 1.6000e+01, 7.2900e+02, 6.5536e+04]]]]);

        assert_eq!(output.dtype(), DType::F32);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn pow_mixed_types() {
        let device = Default::default();
        let model: pow_mixed::Model = pow_mixed::Model::new(&device);
        let f = Tensor::<1>::from_floats([-2.0, 1.5, 3.0, -1.0], &device);
        let i = Tensor::<1, Int>::from_ints([3, 2, 4, 5], &device);
        let fe = Tensor::<1>::from_floats([2.0, 3.0, 0.5, 1.0], &device);
        let u = Tensor::<1, Int>::from_data(
            TensorData::from([2u64, 3, 1, 4]),
            (&device, burn::tensor::DType::U64),
        );

        let (float_int, int_float, float_uint) = model.forward(f, i, fe, u);

        let tolerance = burn::tensor::Tolerance::default();
        assert_eq!(float_int.dtype(), DType::F32);
        float_int
            .to_data()
            .assert_approx_eq::<f32>(&TensorData::from([-8.0f32, 2.25, 81.0, -1.0]), tolerance);
        int_float
            .to_data()
            .assert_eq(&TensorData::from([9i64, 8, 2, 5]), true);
        assert_eq!(float_uint.dtype(), DType::F32);
        float_uint
            .to_data()
            .assert_approx_eq::<f32>(&TensorData::from([4.0f32, 3.375, 3.0, 1.0]), tolerance);

        // A fractional int ^ float result truncates toward zero, as in the onnx reference.
        let (_, int_float, _) = model.forward(
            Tensor::<1>::ones([4], &device),
            Tensor::<1, Int>::from_ints([5, 7, 2, 10], &device),
            Tensor::<1>::from_floats([0.5, 0.5, -1.0, 0.3], &device),
            Tensor::<1, Int>::from_data(
                TensorData::from([1u64, 1, 1, 1]),
                (&device, burn::tensor::DType::U64),
            ),
        );
        int_float
            .to_data()
            .assert_eq(&TensorData::from([2i64, 2, 0, 1]), true);
    }
}
