use crate::include_models;
include_models!(
    inherit_randomnormallike,
    inherit_randomuniformlike,
    seed_randomnormal,
    seed_randomuniform,
    seed_randomnormallike,
    seed_randomuniformlike,
    seed_bernoulli,
    seed_multinomial
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Tensor};
    #[test]
    fn like_operators_inherit_double_dtype() {
        let device = Default::default();
        let normal = inherit_randomnormallike::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/inherit_randomnormallike.bpk"),
            &device,
        );
        let uniform = inherit_randomuniformlike::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/inherit_randomuniformlike.bpk"),
            &device,
        );
        let input = Tensor::<2>::zeros([2, 64], (&device, DType::F64));
        assert_eq!(normal.forward(input.clone()).dtype(), DType::F64);
        assert_eq!(uniform.forward(input).dtype(), DType::F64);
    }

    #[test]
    fn randomnormal_seed_is_reproducible() {
        let device = Default::default();
        let model = seed_randomnormal::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/seed_randomnormal.bpk"),
            &device,
        );
        let (a, b, c) = model.forward();
        let (again, _, _) = model.forward();
        assert_eq!(a.dtype(), DType::F64);
        let a = a.to_data();
        a.assert_eq(&b.to_data(), true);
        a.assert_eq(&again.to_data(), true);
        assert_ne!(
            a,
            c.to_data(),
            "different fractional seeds must produce different samples"
        );
    }
    #[test]
    fn randomuniform_seed_is_reproducible() {
        let device = Default::default();
        let model = seed_randomuniform::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/seed_randomuniform.bpk"),
            &device,
        );
        let (a, b, c) = model.forward();
        let (again, _, _) = model.forward();
        assert_eq!(a.dtype(), DType::F64);
        let a = a.to_data();
        a.assert_eq(&b.to_data(), true);
        a.assert_eq(&again.to_data(), true);
        assert_ne!(
            a,
            c.to_data(),
            "different fractional seeds must produce different samples"
        );
    }
    #[test]
    fn randomnormallike_seed_is_reproducible() {
        let device = Default::default();
        let model = seed_randomnormallike::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/seed_randomnormallike.bpk"),
            &device,
        );
        let input = Tensor::<2>::full([2, 64], 0.5, (&device, DType::F32));
        let (a, b, c) = model.forward(input.clone());
        let (again, _, _) = model.forward(input.clone());
        assert_eq!(a.dtype(), DType::F64);
        let a = a.to_data();
        a.assert_eq(&b.to_data(), true);
        a.assert_eq(&again.to_data(), true);
        assert_ne!(
            a,
            c.to_data(),
            "different fractional seeds must produce different samples"
        );
    }
    #[test]
    fn randomuniformlike_seed_is_reproducible() {
        let device = Default::default();
        let model = seed_randomuniformlike::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/seed_randomuniformlike.bpk"),
            &device,
        );
        let input = Tensor::<2>::full([2, 64], 0.5, (&device, DType::F32));
        let (a, b, c) = model.forward(input.clone());
        let (again, _, _) = model.forward(input.clone());
        assert_eq!(a.dtype(), DType::F64);
        let a = a.to_data();
        a.assert_eq(&b.to_data(), true);
        a.assert_eq(&again.to_data(), true);
        assert_ne!(
            a,
            c.to_data(),
            "different fractional seeds must produce different samples"
        );
    }
    #[test]
    fn bernoulli_seed_is_reproducible() {
        let device = Default::default();
        let model = seed_bernoulli::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/seed_bernoulli.bpk"),
            &device,
        );
        let input = Tensor::<2>::full([2, 64], 0.5, (&device, DType::F32));
        let (a, b, c) = model.forward(input.clone());
        let (again, _, _) = model.forward(input.clone());
        assert_eq!(a.dtype(), DType::I64);
        for probability in [0.0, 1.0] {
            let input = Tensor::<2>::full([2, 64], probability, (&device, DType::F32));
            let (sample, _, _) = model.forward(input);
            assert!(
                sample
                    .to_data()
                    .to_vec::<i64>()
                    .unwrap()
                    .iter()
                    .all(|&x| x == probability as i64)
            );
        }
        let a = a.to_data();
        a.assert_eq(&b.to_data(), true);
        a.assert_eq(&again.to_data(), true);
        assert_ne!(
            a,
            c.to_data(),
            "different fractional seeds must produce different samples"
        );
    }
    #[test]
    fn multinomial_seed_is_reproducible() {
        let device = Default::default();
        let model = seed_multinomial::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/seed_multinomial.bpk"),
            &device,
        );
        let input = Tensor::<2>::full([2, 64], 0.5, (&device, DType::F32));
        let (a, b, c) = model.forward(input.clone());
        let (again, _, _) = model.forward(input.clone());
        assert_eq!(a.dtype(), DType::I64);
        assert!(
            a.to_data()
                .to_vec::<i64>()
                .unwrap()
                .iter()
                .all(|&x| (0..64).contains(&x))
        );
        let mut logits = alloc::vec![f32::NEG_INFINITY;128];
        logits[63] = 0.0;
        logits[127] = 0.0;
        let certain = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(logits, [2, 64]),
            (&device, DType::F32),
        );
        let (sample, _, _) = model.forward(certain);
        assert!(
            sample
                .to_data()
                .to_vec::<i64>()
                .unwrap()
                .iter()
                .all(|&x| x == 63)
        );
        let a = a.to_data();
        a.assert_eq(&b.to_data(), true);
        a.assert_eq(&again.to_data(), true);
        assert_ne!(
            a,
            c.to_data(),
            "different fractional seeds must produce different samples"
        );
    }
}
