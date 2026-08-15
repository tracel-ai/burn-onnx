//! Importing an ONNX model that contains operators burn-onnx does not know.
//!
//! The hooks live in `build.rs`; the kernels they call live in [`ops`].

pub mod model;
pub mod ops;

pub use model::custom_model;
