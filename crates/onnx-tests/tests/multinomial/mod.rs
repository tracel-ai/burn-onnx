// Multinomial draws random samples, so these tests check invariants rather than exact values:
// output shape and dtype, that every index is a valid class, and that a one-hot input (one
// class at log-probability 0, the rest at -inf or -1e9) always samples that class.
use crate::include_models;
include_models!(
    multinomial_half,
    multinomial_empty,
    multinomial,
    multinomial_int64,
    multinomial_default
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, Int, Tensor, TensorData};

    fn input(device: &Device) -> Tensor<2> {
        Tensor::<2>::from_data(
            TensorData::from([
                [f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY],
                [10000.0, -1e9, -1e9, -1e9],
                [0.0, 0.0, 0.0, 0.0],
            ]),
            (device, DType::F32),
        )
    }

    fn check_rows(output: Tensor<2, Int>, sample_size: usize) {
        let rows = output.to_data().convert::<i64>().to_vec::<i64>().unwrap();
        let (one_hot_2, rest) = rows.split_at(sample_size);
        let (one_hot_0, uniform) = rest.split_at(sample_size);
        assert!(one_hot_2.iter().all(|&v| v == 2), "{one_hot_2:?}");
        assert!(one_hot_0.iter().all(|&v| v == 0), "{one_hot_0:?}");
        assert!(uniform.iter().all(|&v| (0..4).contains(&v)), "{uniform:?}");
    }

    #[test]
    fn multinomial_int32_default() {
        let device = Default::default();
        let model = multinomial::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/multinomial.bpk"),
            &device,
        );

        let output = model.forward(input(&device));
        assert_eq!(output.dims(), [3, 16]);
        assert_eq!(output.dtype(), DType::I32);
        check_rows(output, 16);
    }

    #[test]
    fn multinomial_int64_with_seed() {
        let device = Default::default();
        let model = multinomial_int64::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/multinomial_int64.bpk"),
            &device,
        );

        let output = model.forward(input(&device));
        assert_eq!(output.dims(), [3, 8]);
        assert_eq!(output.dtype(), DType::I64);
        check_rows(output, 8);
    }
    #[test]
    fn default_sample_count() {
        let device = Default::default();
        let model = multinomial_default::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/multinomial_default.bpk"),
            &device,
        );
        let output = model.forward(input(&device));
        assert_eq!(output.dims(), [3, 1]);
        assert_eq!(output.dtype(), DType::I32);
        check_rows(output, 1);
    }
}

#[test]
fn multinomial_large_half_and_empty_batch() {
    use burn::tensor::{DType, Tensor, TensorData};
    let device = Default::default();
    let model = multinomial_half::Model::from_file(
        concat!(env!("OUT_DIR"), "/model/multinomial_half.bpk"),
        &device,
    );
    let x = Tensor::<2>::from_data(
        TensorData::new(alloc::vec![0f32;65536], [1, 65536]),
        (&device, DType::F16),
    );
    let samples = model.forward(x).into_data().to_vec::<i32>().unwrap();
    assert!(samples.iter().any(|&v| v < 32768));
    assert!(samples.iter().any(|&v| v >= 32768));
    let model = multinomial_empty::Model::from_file(
        concat!(env!("OUT_DIR"), "/model/multinomial_empty.bpk"),
        &device,
    );
    let x = Tensor::<2>::from_data(
        TensorData::new(alloc::vec::Vec::<f32>::new(), [0, 4]),
        (&device, DType::F32),
    );
    assert_eq!(model.forward(x).dims(), [0, 2]);
}
