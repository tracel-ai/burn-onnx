use crate::include_models;
include_models!(custom_ops);

/// Runtime implementations for the custom ops. The CustomOp hooks in
/// build.rs emit calls into this module from the generated forward().
pub mod ops {
    use burn::prelude::*;

    pub fn scale_shift(x: Tensor<2>, scale: f32, shift: f32) -> Tensor<2> {
        x * scale + shift
    }

    pub fn add_window(x: Tensor<2>, window: &[f32], device: &Device) -> Tensor<2> {
        let window = Tensor::<1>::from_floats(window, device);
        x + window.unsqueeze()
    }

    /// Backs the OpOverride for the built-in Relu. Deliberately deviates
    /// from relu (adds 1.0) so the test can prove the override, not the
    /// built-in codegen, produced the output.
    pub fn my_relu(x: Tensor<2>) -> Tensor<2> {
        x.clamp_min(0.0) + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn custom_ops() {
        // new() is safe here: the AddWindow hook inlines the window values at
        // codegen time via Argument::value(), so the (zero-initialized)
        // constant Param the initializer produces is never consumed.
        let device = Default::default();
        let model: custom_ops::Model = custom_ops::Model::new(&device);

        let input =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]], &device);
        let output = model.forward(input);

        // relu((x * 2.0 + 0.5) + [0.25, 0.5, 0.75, 1.0]) + 1.0; see
        // custom_ops.py. The trailing +1.0 comes from the deliberately
        // unfaithful ReluOverride and proves override dispatch end to end.
        let expected = TensorData::from([[3.75f32, 6.0, 8.25, 10.5], [1.0, 1.0, 1.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }
}
