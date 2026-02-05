//! # ScatterND
//!
//! Updates a copy of the data tensor at positions specified by indices with values from updates.
//! This is the inverse of GatherND.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__ScatterND.html>
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

/// Reduction mode for ScatterND.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ScatterNDReduction {
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

/// Configuration for the ScatterND operation.
#[derive(Debug, Clone, new, Default)]
pub struct ScatterNDConfig {
    pub reduction: ScatterNDReduction,
}

/// Node representation for ScatterND operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct ScatterNDNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ScatterNDConfig,
}

pub(crate) struct ScatterNDProcessor;

impl NodeProcessor for ScatterNDProcessor {
    type Config = ScatterNDConfig;

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
        let mut reduction = ScatterNDReduction::None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "reduction" => {
                    let s = value.clone().into_string();
                    reduction = match s.as_str() {
                        "none" => ScatterNDReduction::None,
                        "add" => ScatterNDReduction::Add,
                        "mul" => ScatterNDReduction::Mul,
                        "max" => ScatterNDReduction::Max,
                        "min" => ScatterNDReduction::Min,
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
                        reason: format!("Unexpected attribute for ScatterND: {}", key),
                    });
                }
            }
        }

        Ok(ScatterNDConfig { reduction })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::ScatterND(ScatterNDNode {
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
        TestNodeBuilder::new(NodeType::ScatterND, "test_scatter_nd")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("indices", 2, None)
            .input_tensor_f32("updates", 1, None)
            .output_tensor_f32("output", 2, None)
    }

    #[test]
    fn test_default_reduction() {
        let node = create_test_node().build();
        let processor = ScatterNDProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.reduction, ScatterNDReduction::None);
    }

    #[test]
    fn test_add_reduction() {
        let node = create_test_node().attr_string("reduction", "add").build();
        let processor = ScatterNDProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.reduction, ScatterNDReduction::Add);
    }

    #[test]
    fn test_mul_reduction() {
        let node = create_test_node().attr_string("reduction", "mul").build();
        let processor = ScatterNDProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.reduction, ScatterNDReduction::Mul);
    }

    #[test]
    fn test_max_reduction() {
        let node = create_test_node().attr_string("reduction", "max").build();
        let processor = ScatterNDProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.reduction, ScatterNDReduction::Max);
    }

    #[test]
    fn test_min_reduction() {
        let node = create_test_node().attr_string("reduction", "min").build();
        let processor = ScatterNDProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.reduction, ScatterNDReduction::Min);
    }

    #[test]
    fn test_invalid_reduction() {
        let node = create_test_node()
            .attr_string("reduction", "invalid")
            .build();
        let processor = ScatterNDProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_unexpected_attribute() {
        let node = create_test_node().attr_int("unknown_attr", 42).build();
        let processor = ScatterNDProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_types() {
        let mut node = create_test_node().build();
        let processor = ScatterNDProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should match data input type
        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.dtype, crate::ir::DType::F32);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }
}
