//! # Logical AND Operation
//!
//! Element-wise logical AND operation with multidirectional broadcasting support.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__And.html>
//!
//! ## Type Constraints
//!
//! T: Boolean tensor types
//!
//! ## Opset Versions
//! - **Opset 1-6**: Limited broadcast support
//! - **Opset 7+**: Multidirectional (Numpy-style) broadcasting

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node representation for And operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct AndNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for logical AND operation
pub(crate) struct AndProcessor;

impl NodeProcessor for AndProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Structural validation for Shape combined with an on-device tensor.
        // burn-onnx codegen materializes the Shape as a rank-1 Int tensor,
        // converts it to bool via `.not_equal_elem(0i64)`, and then calls
        // `bool_and` on the counterpart. That only compiles when the other
        // operand is a rank-1 Bool tensor. Reject anything else here with a
        // clean ProcessError so users don't see a cryptic deep-burn type
        // error at generated-code compile time. ONNX And already constrains
        // T to bool, but an upstream dtype-inference bug could still let a
        // non-bool tensor reach this arm.
        let lhs_ty = &node.inputs[0].ty;
        let rhs_ty = &node.inputs[1].ty;
        if let (ArgType::Shape(_), other) | (other, ArgType::Shape(_)) = (lhs_ty, rhs_ty)
            && other.is_on_device()
        {
            let other_dtype = other.elem_type();
            if !other_dtype.is_bool() {
                return Err(ProcessError::TypeMismatch {
                    expected: "bool tensor when combined with Shape".to_string(),
                    actual: format!("{other_dtype:?}"),
                });
            }
            if other.rank() != 1 {
                return Err(ProcessError::Custom(format!(
                    "And: Shape combined with rank-{} bool tensor is not supported (expected rank 1)",
                    other.rank()
                )));
            }
        }

        same_as_input_broadcast(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::And(AndNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;

    /// And on Shape + rank-2 bool tensor must be rejected. The codegen
    /// arm for this combination assumes rank 1; higher ranks would
    /// miscompile or panic deep in burn.
    #[test]
    fn shape_and_rank2_bool_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::And, "and_shape_rank2")
            .input_shape("lhs", 3)
            .input_tensor_bool("rhs", 2, None)
            .output_tensor_bool("out", 2, None)
            .build();

        let processor = AndProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        match result {
            Err(ProcessError::Custom(msg)) => assert!(
                msg.contains("rank-2") && msg.contains("And"),
                "expected rank-2 message, got: {msg}"
            ),
            other => panic!("expected Custom ProcessError, got {other:?}"),
        }
    }

    /// And on Shape + rank-1 float tensor must be rejected because
    /// codegen only supports the bool counterpart.
    #[test]
    fn shape_and_float_tensor_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::And, "and_shape_float")
            .input_shape("lhs", 3)
            .input_tensor_f32("rhs", 1, None)
            .output_tensor_bool("out", 1, None)
            .build();

        let processor = AndProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::TypeMismatch { .. })),
            "expected TypeMismatch, got {result:?}"
        );
    }

    /// And on Shape + rank-1 bool tensor is the legitimate case and
    /// must be accepted.
    #[test]
    fn shape_and_rank1_bool_accepted() {
        let mut node = TestNodeBuilder::new(NodeType::And, "and_shape_rank1")
            .input_shape("lhs", 3)
            .input_tensor_bool("rhs", 1, None)
            .output_tensor_bool("out", 1, None)
            .build();

        let processor = AndProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_ok(), "expected Ok, got {result:?}");
    }
}
