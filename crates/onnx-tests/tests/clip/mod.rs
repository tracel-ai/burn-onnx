// Import the shared macro
use crate::include_models;
include_models!(clip, clip_int_static_min_runtime_max);

// Runtime bounds go through a separate codegen path. Deny `unused_parens` so a
// regression in the generated bound casts fails to compile instead of warning.
#[deny(unused_parens)]
pub mod clip_runtime_bounds {
    include!(concat!(env!("OUT_DIR"), "/model/clip_runtime_bounds.rs"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn clip_int_static_min_runtime_max() {
        let device = Default::default();
        let model = clip_int_static_min_runtime_max::Model::from_file(
            concat!(
                env!("OUT_DIR"),
                "/model/clip_int_static_min_runtime_max.bpk"
            ),
            &device,
        );
        let input = Tensor::<1, burn::tensor::Int>::from_data(
            TensorData::from([0i64, 3, 9]),
            (&device, burn::tensor::DType::I64),
        );
        let output = model.forward(input, 5);
        output
            .to_data()
            .assert_eq(&TensorData::from([1i64, 3, 5]), true);
    }

    #[test]
    fn clip() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: clip::Model = clip::Model::new(&device);

        // Run the model
        let input = Tensor::<1>::from_floats(
            [
                0.88226926,
                0.91500396,
                0.38286376,
                0.95930564,
                0.390_448_2,
                0.60089535,
            ],
            &device,
        );
        let (output1, output2, output3) = model.forward(input);
        let expected1 = TensorData::from([
            0.88226926f32,
            0.91500396,
            0.38286376,
            0.95930564,
            0.390_448_2,
            0.60089535,
        ]);
        let expected2 = TensorData::from([0.7f32, 0.7, 0.5, 0.7, 0.5, 0.60089535]);
        let expected3 = TensorData::from([0.8f32, 0.8, 0.38286376, 0.8, 0.390_448_2, 0.60089535]);

        output1.to_data().assert_eq(&expected1, true);
        output2.to_data().assert_eq(&expected2, true);
        output3.to_data().assert_eq(&expected3, true);
    }

    #[test]
    fn clip_runtime_bounds() {
        // Expected values from clip_runtime_bounds.py (onnx ReferenceEvaluator).
        let device = Default::default();
        let model = clip_runtime_bounds::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/clip_runtime_bounds.bpk"),
            &device,
        );
        let x = Tensor::<1>::from_floats([-2.0, -0.5, 0.5, 2.0], &device);
        let x_i = Tensor::<1, burn::tensor::Int>::from_ints([-5, -1, 1, 5], &device);

        let (both_f, min_only_f, max_only_i, both_i) = model.forward(x, -1.0, 1.0, x_i, -2, 3);

        both_f
            .to_data()
            .assert_eq(&TensorData::from([-1.0f32, -0.5, 0.5, 1.0]), true);
        min_only_f
            .to_data()
            .assert_eq(&TensorData::from([-1.0f32, -0.5, 0.5, 2.0]), true);
        max_only_i
            .to_data()
            .assert_eq(&TensorData::from([-5i32, -1, 1, 3]), true);
        both_i
            .to_data()
            .assert_eq(&TensorData::from([-2i32, -1, 1, 3]), true);
    }
}
