//! # Softmax
//!
//! Applies the Softmax activation function along a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Softmax.html>
//!
//! ## Type Constraints
//! - T: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with axis=1 default, operates on 2D tensors.
//! - **Opset 11**: Allows negative axes and retains 2D coercion with default axis 1.
//! - **Opset 13**: Removed 2D coercion behavior. Softmax now operates along specified axis directly without reshaping. This is the current behavior.
//!
//! Inputs are coerced to 2D before opset 13. Later opsets normalize along the specified axis.

use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for Softmax operations
#[derive(Debug, Clone, new)]
pub struct SoftmaxConfig {
    /// Axis along which to apply softmax
    pub axis: usize,
    /// Coerce the input to two dimensions at axis, as required before opset 13.
    #[new(value = "false")]
    pub flatten: bool,
}

/// Node representation for Softmax operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SoftmaxNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SoftmaxConfig,
}

pub(crate) struct SoftmaxProcessor;

impl NodeProcessor for SoftmaxProcessor {
    type Config = SoftmaxConfig;

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
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // The spec requires the input rank to be >= 1 for the axis attribute to be valid.
        let rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor with rank >= 1".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        if rank == 0 {
            return Err(ProcessError::TypeMismatch {
                expected: "Tensor with rank >= 1".to_string(),
                actual: "Tensor with rank 0 (scalar)".to_string(),
            });
        }

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract the shape of the input tensor
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) if tensor.rank > 0 => tensor.clone(),
            ArgType::Tensor(tensor) => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor with rank >= 1".to_string(),
                    actual: format!("Tensor with rank {}", tensor.rank),
                });
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor with rank >= 1".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Axis default changed between opset versions:
        // opset 1-12: default axis = 1 (with 2D coercion semantics)
        // opset 13+: default axis = -1 (direct axis operation)
        let mut axis: i64 = if opset < 13 { 1 } else { -1 };

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = value.clone().into_i64()?
            }
        }

        // Validate axis is in valid range [-rank, rank-1]
        let rank_i64 = tensor.rank as i64;
        if axis < -rank_i64 || axis >= rank_i64 {
            return Err(ProcessError::InvalidAttribute {
                name: "axis".to_string(),
                reason: format!(
                    "axis {} is out of range for input rank {}; expected in [{}, {}]",
                    axis,
                    tensor.rank,
                    -rank_i64,
                    rank_i64 - 1
                ),
            });
        }

        // if axis is negative, it is counted from the end
        if axis < 0 {
            axis += rank_i64;
        }

        let config = SoftmaxConfig {
            axis: axis as usize,
            flatten: opset < 13,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::Softmax(SoftmaxNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize) -> RawNode {
        TestNodeBuilder::new(NodeType::Softmax, "test_softmax")
            .input_tensor_f32("data", input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_softmax_config_basic() {
        let mut node = create_test_node(-1, 3);
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.axis, 2); // -1 + 3 = 2 (last dimension)
    }

    #[test]
    fn test_softmax_config_explicit_axis() {
        let mut node = create_test_node(1, 3);
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_softmax_config_multiple_inputs() {
        let mut node = create_test_node(1, 3);
        // Add an extra input
        let extra_input = TestNodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);
        let processor = SoftmaxProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_softmax_rank_zero_rejected_infer_types() {
        // Rank-0 (scalar) input should be rejected in infer_types
        let mut node = create_test_node(0, 0);
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::TypeMismatch { .. })),
            "Expected TypeMismatch for rank-0 input, got: {:?}",
            result
        );
    }

    #[test]
    fn test_softmax_rank_zero_rejected_extract_config() {
        // Rank-0 (scalar) input should also be rejected in extract_config
        let node = create_test_node(0, 0);
        let processor = SoftmaxProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(
            matches!(result, Err(ProcessError::TypeMismatch { .. })),
            "Expected TypeMismatch for rank-0 input, got: {:?}",
            result
        );
    }

    #[test]
    fn test_softmax_axis_out_of_range() {
        // axis=5 for rank-3 tensor should be rejected
        let node = create_test_node(5, 3);
        let processor = SoftmaxProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(
            matches!(
                result,
                Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "axis"
            ),
            "Expected InvalidAttribute error for out-of-range axis, got: {:?}",
            result
        );
    }

    #[test]
    fn test_softmax_axis_negative_out_of_range() {
        // axis=-4 for rank-3 tensor should be rejected
        let node = create_test_node(-4, 3);
        let processor = SoftmaxProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(
            matches!(
                result,
                Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "axis"
            ),
            "Expected InvalidAttribute error for negative out-of-range axis, got: {:?}",
            result
        );
    }

    // TODO: Missing test for type constraints - Softmax only supports float types.
    // Need test to verify integer input is rejected (or properly handled).

    #[test]
    fn test_softmax_1d_axis_zero() {
        // Simplest valid case: 1D tensor with axis=0
        let mut node = create_test_node(0, 1);
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.axis, 0);
    }

    #[test]
    fn legacy_coercion_depends_on_opset() {
        let processor = SoftmaxProcessor;
        let node = create_test_node(-1, 3);
        for opset in [11, 12, 13, 18] {
            let config = processor.extract_config(&node, opset).unwrap();
            assert_eq!(config.flatten, opset < 13);
        }
    }
}
