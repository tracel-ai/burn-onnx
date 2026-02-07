//! # Softsign Operation
//!
//! Element-wise softsign activation operation: y = x / (1 + |x|).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Softsign.html>
//!
//! ## Type Constraints
//!
//! T: Float tensor types
//!
//! ## Opset Versions
//! - **Opset 1**: Initial support

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};
use onnx_ir_derive::NodeBuilder;

/// Node representation for Softsign operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SoftsignNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for softsign operation
pub(crate) struct SoftsignProcessor;

impl NodeProcessor for SoftsignProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 1)?;
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Softsign(SoftsignNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
