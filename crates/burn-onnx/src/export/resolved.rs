//! Exporter intermediate representation produced by shape resolution.
//!
//! This representation is the contract between graph capture/shape analysis
//! and ONNX lowering. Lowering consumes explicit shape expressions and does not
//! need to know whether they came from static, paired-trace, or future symbolic
//! resolution.

use burn::backend::ir::{GraphIr, TensorId};

/// Symbolic axis attached to a captured runtime input or graph output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DynamicAxis {
    /// Tensor carrying the symbolic dimension.
    pub(crate) tensor: TensorId,
    /// Axis within the tensor.
    pub(crate) axis: usize,
    /// ONNX symbolic dimension name.
    pub(crate) symbol: String,
}

/// An explicit ONNX-compatible dimension expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShapeExpr {
    /// Constant dimension.
    Static(usize),
    /// Dimension of a declared runtime input.
    InputDim {
        /// Captured runtime input tensor.
        input: TensorId,
        /// Axis within the input tensor.
        axis: usize,
    },
    /// Dimension of an intermediate or source tensor.
    TensorDim {
        /// Captured tensor providing the dimension.
        tensor: TensorId,
        /// Axis within the source tensor.
        axis: usize,
    },
    /// Element-count-preserving inferred dimension (`-1` in ONNX reshape).
    Infer,
}

/// Resolved shape operand for one shape-sensitive operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedShape {
    /// Operation index in [`GraphIr::operations`].
    pub(crate) operation: usize,
    /// Output tensor receiving the shape.
    pub(crate) tensor: TensorId,
    /// Dimension expressions in axis order.
    pub(crate) dimensions: Vec<ShapeExpr>,
}

/// Captured graph plus the explicit shape information required by lowering.
///
/// The graph has already passed the structural checks appropriate to its shape
/// resolver. `shapes` contains runtime expressions for shape operands such as
/// reshape targets, while `dynamic_axes` controls symbolic ONNX boundary
/// dimensions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedExportGraph {
    /// Validated captured graph.
    pub(crate) graph: GraphIr,
    /// Resolved shape-sensitive operands.
    pub(crate) shapes: Vec<ResolvedShape>,
    /// Symbolic dimensions declared on graph boundaries.
    pub(crate) dynamic_axes: Vec<DynamicAxis>,
}
