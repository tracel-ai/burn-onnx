use crate::include_models;
include_models!(pow, pow_int, pow_broadcast);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn pow_int_with_tensor_and_scalar() {
        let device = Default::default();
        let model: pow_int::Model<TestBackend> = pow_int::Model::new(&device);

        let input1 = Tensor::<TestBackend, 4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let input2 = 2;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[[[1i64, 16, 729, 65536]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn pow_broadcast() {
        let device = Default::default();
        let model: pow_broadcast::Model<TestBackend> = pow_broadcast::Model::new(&device);

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
        let model: pow::Model<TestBackend> = pow::Model::new(&device);

        let input1 = Tensor::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let input2 = 2f64;

        let output = model.forward(input1, input2);

        let expected = TensorData::from([[[[1.0000f32, 1.6000e+01, 7.2900e+02, 6.5536e+04]]]]);

        output.to_data().assert_eq(&expected, true);
    }
}
