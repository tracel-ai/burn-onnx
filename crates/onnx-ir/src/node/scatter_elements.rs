//! # ScatterElements
//!
//! Updates a copy of the data tensor with values from `updates` at positions specified by `indices`
//! along a given axis. This is the inverse of GatherElements.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__ScatterElements.html>
//!
//! ## Opset Versions
//! - **Opset 11**: Initial version.
//! - **Opset 13**: Clarifications, bfloat16 support.
//! - **Opset 16**: Added add/mul reduction.
//! - **Opset 18**: Added max/min reduction.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Reduction mode for ScatterElements.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ScatterElementsReduction {
    /// No reduction (default). Duplicate indices are undefined behavior.
    #[default]
    None,
    /// Reduction using addition.
    Add,
    /// Reduction using multiplication.
    Mul,
    /// Reduction using maximum.
    Max,
    /// Reduction using minimum.
    Min,
}

/// Configuration for the ScatterElements operation.
#[derive(Debug, Clone, new)]
pub struct ScatterElementsConfig {
    pub axis: usize,
    pub reduction: ScatterElementsReduction,
}

/// Node representation for ScatterElements operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct ScatterElementsNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ScatterElementsConfig,
}

pub(crate) struct ScatterElementsProcessor;

impl NodeProcessor for ScatterElementsProcessor {
    type Config = ScatterElementsConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Output has same type and shape as data input
        if let crate::ir::ArgType::Tensor(data_tensor) = &node.inputs[0].ty {
            node.outputs[0].ty = crate::ir::ArgType::Tensor(data_tensor.clone());
        }
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let input_dim = match &node.inputs[0].ty {
            crate::ir::ArgType::Tensor(tensor) => tensor.rank as i64,
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", other),
                });
            }
        };

        let mut axis: i64 = 0;
        let mut reduction = ScatterElementsReduction::None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                "reduction" => {
                    let s = value.clone().into_string();
                    reduction = match s.as_str() {
                        "none" => ScatterElementsReduction::None,
                        "add" => ScatterElementsReduction::Add,
                        "mul" => ScatterElementsReduction::Mul,
                        "max" => ScatterElementsReduction::Max,
                        "min" => ScatterElementsReduction::Min,
                        _ => {
                            return Err(ProcessError::InvalidAttribute {
                                name: "reduction".to_string(),
                                reason: format!(
                                    "Invalid reduction mode '{}'. Must be one of: none, add, mul, max, min",
                                    s
                                ),
                            });
                        }
                    };
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for ScatterElements: {}", key),
                    });
                }
            }
        }

        // Normalize negative axis
        if axis < 0 {
            axis += input_dim;
        }

        if axis < 0 || axis >= input_dim {
            return Err(ProcessError::InvalidAttribute {
                name: "axis".to_string(),
                reason: format!("axis {} is out of bounds for rank {}", axis, input_dim),
            });
        }

        Ok(ScatterElementsConfig {
            axis: axis as usize,
            reduction,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::ScatterElements(ScatterElementsNode {
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
    use crate::processor::OutputPreferences;

    fn create_test_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::ScatterElements, "test_scatter_elements")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("indices", 2, None)
            .input_tensor_f32("updates", 2, None)
            .output_tensor_f32("output", 2, None)
    }

    #[test]
    fn test_default_config() {
        let node = create_test_node().build();
        let processor = ScatterElementsProcessor;
        let config = processor.extract_config(&node, 11).unwrap();
        assert_eq!(config.axis, 0);
        assert_eq!(config.reduction, ScatterElementsReduction::None);
    }

    #[test]
    fn test_axis_1() {
        let node = create_test_node().attr_int("axis", 1).build();
        let processor = ScatterElementsProcessor;
        let config = processor.extract_config(&node, 11).unwrap();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_negative_axis() {
        let node = create_test_node().attr_int("axis", -1).build();
        let processor = ScatterElementsProcessor;
        let config = processor.extract_config(&node, 11).unwrap();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_axis_out_of_bounds() {
        let node = create_test_node().attr_int("axis", 2).build();
        let processor = ScatterElementsProcessor;
        let result = processor.extract_config(&node, 11);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_reduction() {
        let node = create_test_node().attr_string("reduction", "add").build();
        let processor = ScatterElementsProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.reduction, ScatterElementsReduction::Add);
    }

    #[test]
    fn test_mul_reduction() {
        let node = create_test_node().attr_string("reduction", "mul").build();
        let processor = ScatterElementsProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.reduction, ScatterElementsReduction::Mul);
    }

    #[test]
    fn test_max_reduction() {
        let node = create_test_node().attr_string("reduction", "max").build();
        let processor = ScatterElementsProcessor;
        let config = processor.extract_config(&node, 18).unwrap();
        assert_eq!(config.reduction, ScatterElementsReduction::Max);
    }

    #[test]
    fn test_min_reduction() {
        let node = create_test_node().attr_string("reduction", "min").build();
        let processor = ScatterElementsProcessor;
        let config = processor.extract_config(&node, 18).unwrap();
        assert_eq!(config.reduction, ScatterElementsReduction::Min);
    }

    #[test]
    fn test_invalid_reduction() {
        let node = create_test_node()
            .attr_string("reduction", "invalid")
            .build();
        let processor = ScatterElementsProcessor;
        let result = processor.extract_config(&node, 18);
        assert!(result.is_err());
    }

    #[test]
    fn test_unexpected_attribute() {
        let node = create_test_node().attr_int("unknown_attr", 42).build();
        let processor = ScatterElementsProcessor;
        let result = processor.extract_config(&node, 11);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_types() {
        let mut node = create_test_node().build();
        let processor = ScatterElementsProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 11, &prefs).unwrap();

        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.dtype, crate::ir::DType::F32);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }
}
