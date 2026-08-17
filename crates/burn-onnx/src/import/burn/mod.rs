/// The graph module.
pub mod graph;

mod codegen;
pub(crate) mod custom_op;
pub(crate) mod node_codegen; // Implements NodeCodegen<PS> for onnx_ir::Node
pub(crate) mod node_traits;

mod imports;

mod argument_helpers;
mod partition;
mod scope;

pub(crate) use argument_helpers::*;
pub(crate) use codegen::ToTokens;
pub(crate) use imports::*;
pub(crate) use node_traits::{Field, TensorKind};
pub(crate) use scope::*;

pub(crate) mod node;
