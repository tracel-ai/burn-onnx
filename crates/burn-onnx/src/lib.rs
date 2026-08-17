#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Import and export ONNX models with Burn.
//!
//! The `burn_onnx::import` module converts ONNX models into Burn Rust source code and
//! Burnpack weight files. Its public API is also re-exported from the crate root
//! for backwards compatibility.
//!
//! The optional `burn_onnx::export` module captures a Burn module's forward graph and
//! serializes it as an ONNX model. Enable the `export` feature to use it.

#[cfg(feature = "import")]
#[macro_use]
extern crate derive_new;

/// ONNX-to-Burn import and code generation.
#[cfg(feature = "import")]
#[cfg_attr(docsrs, doc(cfg(feature = "import")))]
pub mod import;

#[cfg(feature = "import")]
pub use import::*;

/// Burn-to-ONNX graph capture and export.
#[cfg(feature = "export")]
#[cfg_attr(docsrs, doc(cfg(feature = "export")))]
pub mod export;
