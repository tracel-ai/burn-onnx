#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deprecated(
    since = "0.21.0",
    note = "burn-import ONNX functionality has moved to burn-onnx. Please migrate to burn-onnx directly."
)]

//! # burn-import (Legacy)
//!
//! **This crate is deprecated.** ONNX import functionality has been moved to
//! [`burn-onnx`](https://crates.io/crates/burn-onnx).
//!
//! PyTorch and Safetensors model weights import has been replaced by
//! [`burn-store`](https://crates.io/crates/burn-store).
//! Check out the [migration guide](https://github.com/tracel-ai/burn/blob/main/crates/burn-store/MIGRATION.md).
//!
//! ## Migration Guide
//!
//! Replace your dependency:
//!
//! ```toml
//! # Before
//! burn-import = { version = "0.21", features = ["onnx"] }
//!
//! # After
//! burn-onnx = "0.21"
//! ```
//!
//! Update your imports:
//!
//! ```ignore
//! // Before
//! use burn_import::onnx::ModelGen;
//!
//! // After
//! use burn_onnx::ModelGen;
//! ```
//!
//! For more information, see the [burn-onnx documentation](https://docs.rs/burn-onnx).

/// The onnx module (deprecated).
///
/// Use [`burn_onnx`] directly instead.
#[deprecated(
    since = "0.21.0",
    note = "Use burn_onnx directly instead of burn_import::onnx"
)]
pub mod onnx {
    pub use burn_onnx::*;
}

/// The burn module for code generation (deprecated).
///
/// Use [`burn_onnx::burn`] directly instead.
#[deprecated(
    since = "0.21.0",
    note = "Use burn_onnx::burn directly instead of burn_import::burn"
)]
pub mod burn {
    pub use burn_onnx::burn::*;
}

// Re-export main types at crate root for convenience during migration
#[doc(hidden)]
#[deprecated(since = "0.21.0", note = "Use burn_onnx::ModelGen instead")]
pub use burn_onnx::ModelGen;
