//! Native regression gate against the upstream ONNX backend test suite.
//!
//! This crate vendors a subset of the upstream ONNX backend node tests
//! (under `vendor/node/`) and runs them against `burn-onnx`-generated
//! Rust code on every `cargo test`.
//!
//! Each upstream test directory follows this layout:
//!
//! ```text
//! vendor/node/test_<name>/
//!   model.onnx
//!   test_data_set_0/
//!     input_0.pb     (optionally input_1.pb, ...)
//!     output_0.pb
//! ```
//!
//! `build.rs` runs `burn_onnx::ModelGen` over every vendored `model.onnx`,
//! emitting one `<name>.rs` + `<name>.bpk` per test in `$OUT_DIR/model/`.
//! The integration test in `tests/test_mod.rs` then loads the `.pb`
//! reference tensors and compares the model output element-wise.
//!
//! See `expectations.toml` for the per-test status declarations.

pub mod expectations;
pub mod pb_loader;
