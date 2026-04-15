#[macro_use]
extern crate derive_new;

mod external_data;
mod graph_state;
pub mod ir;
pub mod node;
mod phases;
mod pipeline;
mod processor;
mod proto_conversion;
mod protos;
mod registry;
mod simplify;
mod tensor_store;

// Public API - only expose essentials
pub use ir::*;
pub use node::*;
pub use pipeline::{Error, OnnxGraphBuilder};

/// Generated protobuf bindings for the ONNX wire format messages that
/// sibling crates need to decode on-disk artifacts.
///
/// Re-exported so callers (e.g. `onnx-official-tests`) can read `.pb`
/// reference tensors and inspect `model.onnx` input/output signatures
/// without rebuilding the bindings themselves.
///
/// * `TensorProto` decodes individual tensor `.pb` reference files.
/// * `ModelProto` / `GraphProto` / `ValueInfoProto` / `TypeProto` /
///   `TensorShapeProto` are used by `onnx-official-tests`'s build script
///   to walk a `model.onnx` header and extract per-test input/output
///   shape and dtype metadata so it can emit per-test harness glue with
///   the correct rank and element type.
///
/// The inner namespaces (`type_proto::Tensor`, `tensor_shape_proto::
/// Dimension`, etc.) stay private. Callers can still reach them via
/// method calls on values of the re-exported outer types — Rust allows
/// public method calls on references to private types as long as the
/// private type is never named in user code.
pub use protos::{
    GraphProto, ModelProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto,
};
