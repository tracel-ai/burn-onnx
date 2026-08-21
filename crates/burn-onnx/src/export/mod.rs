//! Export captured Burn operation graphs to ONNX.
//!
//! Shape validation and resolution deliberately precede ONNX lowering. This
//! keeps trace-based inference replaceable by a future symbolic capture pass.

mod error;
mod exporter;
mod lower;
mod model;
mod resolved;
mod shape;
mod validate;

pub use error::ExportError;
#[doc(hidden)]
pub use exporter::{ExportInput, ExportOutput};
pub use exporter::{OnnxExporter, Opset};
pub use model::OnnxModel;
pub(crate) use resolved::{DynamicAxis, ResolvedExportGraph, ResolvedShape, ShapeExpr};
pub use shape::{AxisSpec, InputSpec};
pub(crate) use validate::GraphStructureValidator;
