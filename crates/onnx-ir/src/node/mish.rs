//! # Mish Activation
//!
//! Mish: A Self Regularized Non-Monotonic Neural Activation Function.
//!
//! `mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))`
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Mish.html>
//!
//! ## Type Constraints
//!
//! T: Float tensor types (float16, float, double, bfloat16)
//!
//! ## Opset Versions
//! - **Opset 18**: Initial support
//! - **Opset 22**: Added bfloat16

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};

/// Node representation for Mish operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MishNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for Mish activation
pub(crate) struct MishProcessor;

impl NodeProcessor for MishProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 18,
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
        validate_opset(opset, 18)?;
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Mish(MishNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
