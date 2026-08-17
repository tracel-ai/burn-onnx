//! Convert ONNX models into Burn Rust source code and Burnpack weight files.

mod logger;

/// Burn code generation.
pub mod burn;

/// Extension traits used by generated Burn models.
pub mod ext;

mod formatter;
mod model_gen;

pub use formatter::*;
pub use model_gen::*;
