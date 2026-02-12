//! # Hardmax
//!
//! Computes the hardmax values for the given input: 1 for the first maximum
//! value along the specified axis, 0 otherwise.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Hardmax.html>
//!
//! ## Type Constraints
//! - T: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with axis=1 default, operates on 2D coercion.
//! - **Opset 11**: Changed default axis behavior.
//! - **Opset 13**: Removed 2D coercion. Hardmax operates along specified axis directly.
//!
//! **Implementation Note**: Requires opset 13+ (no 2D coercion). Default axis is -1.

use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for Hardmax operations
#[derive(Debug, Clone, new)]
pub struct HardmaxConfig {
    /// Axis along which to compute hardmax
    pub axis: usize,
}

/// Node representation for Hardmax operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct HardmaxNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: HardmaxConfig,
}

pub(crate) struct HardmaxProcessor;

impl NodeProcessor for HardmaxProcessor {
    type Config = HardmaxConfig;

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
        crate::processor::same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, opset: usize) -> Result<Self::Config, ProcessError> {
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Axis default: 1 for opset < 13, -1 for opset 13+
        let mut axis: i64 = if opset < 13 { 1 } else { -1 };
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = value.clone().into_i64();
            }
        }

        if axis < 0 {
            axis += tensor.rank as i64;
        }

        Ok(HardmaxConfig {
            axis: axis as usize,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Hardmax(HardmaxNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize) -> RawNode {
        TestNodeBuilder::new(NodeType::Hardmax, "test_hardmax")
            .input_tensor_f32("data", input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_hardmax_config_default_axis() {
        let node = create_test_node(-1, 3);
        let processor = HardmaxProcessor;
        let config = processor.extract_config(&node, 13).unwrap();
        assert_eq!(config.axis, 2); // -1 + 3 = 2
    }

    #[test]
    fn test_hardmax_config_explicit_axis() {
        let node = create_test_node(1, 3);
        let processor = HardmaxProcessor;
        let config = processor.extract_config(&node, 13).unwrap();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_hardmax_infer_types() {
        let mut node = create_test_node(1, 3);
        let processor = HardmaxProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 13, &prefs).unwrap();
        assert_eq!(node.inputs[0].ty, node.outputs[0].ty);
    }
}
