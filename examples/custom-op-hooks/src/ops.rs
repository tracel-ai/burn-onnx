//! Runtime implementations behind the hooks in `build.rs`.
//!
//! The hooks only emit *calls*; these functions are the actual math. They are
//! ordinary Rust, so they can be tested, benchmarked, and swapped for a fused
//! or hardware-specific kernel without touching the ONNX import.

use burn::prelude::*;

/// Backs `example.custom::ScaleBias`: `y = x * scale + bias`.
///
/// The scalars come from ONNX attributes, so the hook inlines them as literals
/// at code generation time; nothing is read from the model at runtime.
pub fn scale_bias(x: Tensor<2>, scale: f32, bias: f32) -> Tensor<2> {
    x * scale + bias
}

/// Backs the `Sigmoid` override.
///
/// A real override would call a fused or vendor kernel here; this one just
/// shows that the built-in codegen was replaced.
pub fn fast_sigmoid(x: Tensor<2>) -> Tensor<2> {
    burn::tensor::activation::sigmoid(x)
}

/// Backs `example.custom::ChannelScale`: scales the last dimension.
///
/// `scale` arrives as a literal slice because the hook read the constant
/// initializer at code generation time via `Argument::value()`.
pub fn channel_scale(x: Tensor<2>, scale: &[f32], device: &Device) -> Tensor<2> {
    let scale = Tensor::<1>::from_floats(scale, device);
    x * scale.unsqueeze()
}
