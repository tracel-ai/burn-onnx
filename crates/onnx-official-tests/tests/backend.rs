//! Test backend selection. Mirrors `crates/onnx-tests/tests/backend.rs`
//! so the two suites stay configurable through the same feature flags.
//!
//! There is no compiler-enforced link between this file and the one in
//! `onnx-tests`. If a new backend feature is added there, copy the
//! corresponding `cfg` block here too.

#[cfg(feature = "test-wgpu")]
pub type TestBackend = burn::backend::Wgpu;

#[cfg(all(
    feature = "test-flex",
    not(feature = "test-wgpu"),
    not(feature = "test-tch"),
    not(feature = "test-metal"),
    not(feature = "test-candle")
))]
pub type TestBackend = burn::backend::Flex;

#[cfg(feature = "test-metal")]
pub type TestBackend = burn::backend::Metal;

#[cfg(feature = "test-tch")]
pub type TestBackend = burn::backend::LibTorch<f32>;

#[cfg(feature = "test-candle")]
pub type TestBackend = burn::backend::Candle<f32>;
