//! # Element-wise Operations
//!
//! Shared node types for element-wise binary operations.

use crate::ir::Argument;

/// Node representation for element-wise binary operations
#[derive(Debug, Clone)]
pub struct ElementwiseBinaryNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub node_type: crate::ir::NodeType,
}
