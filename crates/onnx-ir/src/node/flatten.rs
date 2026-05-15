//! # Flatten
//!
//! Flattens input tensor into a 2D matrix by splitting at a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Flatten.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic flatten operation.
//! - **Opset 9**: No functional changes (extended type support).
//! - **Opset 11**: Added support for negative axis values.
//! - **Opset 13**: Extended type constraints (added bfloat16 support).
//!
//! **Implementation Note**: Negative axis is rejected for opset < 11.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for Flatten operations
#[derive(Debug, Clone, new)]
pub struct FlattenConfig {
    /// Axis along which to flatten
    pub axis: usize,
}

/// Node representation for Flatten operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct FlattenNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: FlattenConfig,
}

pub(crate) struct FlattenProcessor;

impl NodeProcessor for FlattenProcessor {
    type Config = FlattenConfig;

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
        // Extract the shape of the input tensor
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Burn cannot represent Tensor<0>, so reject rank-0 scalars.
        // Rank-1 is valid per ONNX spec: axis=0 -> [1, d], axis=1 -> [d, 1].
        if tensor.rank == 0 {
            return Err(ProcessError::Custom(
                "Flatten: Burn requires input rank >= 1 (got 0)".to_string(),
            ));
        }

        // Reject negative axis for opset < 11 (negative axis was added in opset 11).
        let raw_axis: i64 = node
            .attrs
            .get("axis")
            .map(|v| v.clone().into_i64())
            .unwrap_or(1);
        if raw_axis < 0 && opset < 11 {
            return Err(ProcessError::Custom(format!(
                "Flatten: negative axis ({}) requires opset >= 11, got opset {}",
                raw_axis, opset
            )));
        }

        // Get reference to config for type inference (normalizes axis).
        let config = self.extract_config(node, opset)?;

        // Validate axis is within valid range [0, rank] after normalization.
        // ONNX spec allows axis in [-r, r] inclusive; after normalization that is [0, rank].
        if config.axis > tensor.rank {
            return Err(ProcessError::Custom(format!(
                "Flatten: axis {} (raw: {}) out of range for input rank {} (valid range: [0, {}])",
                config.axis, raw_axis, tensor.rank, tensor.rank
            )));
        }

        // Compute output static_shape: [product(dims[..axis]), product(dims[axis..])]
        let static_shape = if let Some(input_shape) = &tensor.static_shape {
            let axis = config.axis;
            let left = input_shape[..axis]
                .iter()
                .try_fold(1usize, |acc, dim| dim.map(|d| acc * d));
            let right = input_shape[axis..]
                .iter()
                .try_fold(1usize, |acc, dim| dim.map(|d| acc * d));
            Some(vec![left, right])
        } else {
            Some(vec![None, None])
        };

        // Infer output type - Flatten to a 2D tensor
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: 2,
            static_shape,
        });

        Ok(())
    }

    fn is_noop(&self, node: &RawNode) -> bool {
        // Flatten always produces rank 2. It's a no-op when input is rank 2 AND axis=1
        // (the default), which splits dimensions as [d0, d1] -> [d0, d1] (identity).
        // axis=0 would give [1, d0*d1] and axis=2 would give [d0*d1, 1].
        if let ArgType::Tensor(in_t) = &node.inputs[0].ty {
            if in_t.rank != 2 {
                return false;
            }
            let axis = node
                .attrs
                .get("axis")
                .map(|v| v.clone().into_i64())
                .unwrap_or(1);
            return axis == 1 || axis == -(in_t.rank as i64 - 1);
        }
        false
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract the shape of the input tensor
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Extract the axis attribute (default: 1 per ONNX spec)
        let mut axis: i64 = 1;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Flatten: {}", key),
                    });
                }
            }
        }

        // if axis is negative, it is counted from the end
        if axis < 0 {
            axis += tensor.rank as i64;
        }

        let config = FlattenConfig {
            axis: axis as usize,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Flatten(FlattenNode {
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

    fn create_test_node(axis: i64) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Flatten, "test_flatten")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", axis)
    }

    #[test]
    fn test_flatten_config_basic() {
        let node = create_test_node(1).process(FlattenProcessor, 16);
        let processor = FlattenProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_flatten_config_with_negative_axis() {
        let node = create_test_node(-2).process(FlattenProcessor, 16);
        let processor = FlattenProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, 2); // -2 + 4 = 2
    }

    #[test]
    fn test_flatten_config_with_rank0_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test_flatten")
            .add_input(
                "x",
                ArgType::Tensor(TensorType {
                    dtype: crate::ir::DType::F32,
                    rank: 0,
                    static_shape: Some(vec![]),
                }),
            )
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 0)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("rank >= 1")));
    }

    #[test]
    fn test_flatten_rank1_invalid_axis_too_large() {
        // axis=2 is out of range for rank-1 input (valid: 0, 1)
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 1, Some(vec![5]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 2)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("out of range"))
        );
    }

    #[test]
    fn test_flatten_rank1_invalid_axis_too_negative() {
        // axis=-2 is out of range for rank-1 input (valid: -1, 0, 1)
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 1, Some(vec![5]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", -2)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("out of range"))
        );
    }

    #[test]
    fn test_flatten_rank1_axis0() {
        // Input [5], axis=0 -> output [1, 5]
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 1, Some(vec![5]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 0)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(1), Some(5)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_flatten_rank1_axis1() {
        // Input [5], axis=1 -> output [5, 1]
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 1, Some(vec![5]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 1)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(5), Some(1)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_flatten_rank1_negative_axis() {
        // Input [5], axis=-1 -> normalized axis=0 -> output [1, 5]
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 1, Some(vec![5]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", -1)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(1), Some(5)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_flatten_config_with_multiple_inputs() {
        let mut node = create_test_node(1).build();
        // Add an extra input
        let extra_input = TestNodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);

        let processor = FlattenProcessor;
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
    fn test_flatten_rank2_axis1_is_noop() {
        let node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 2, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 1)
            .build();
        assert!(FlattenProcessor.is_noop(&node));
    }

    #[test]
    fn test_flatten_rank2_axis_neg1_is_noop() {
        let node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 2, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", -1)
            .build();
        assert!(FlattenProcessor.is_noop(&node));
    }

    #[test]
    fn test_flatten_rank2_axis0_is_not_noop() {
        let node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 2, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 0)
            .build();
        assert!(!FlattenProcessor.is_noop(&node));
    }

    #[test]
    fn test_flatten_rank3_axis1_is_not_noop() {
        let node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 1)
            .build();
        assert!(!FlattenProcessor.is_noop(&node));
    }

    #[test]
    fn test_flatten_negative_axis_rejected_before_opset_11() {
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", -1)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 10, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("opset >= 11"))
        );
    }

    #[test]
    fn test_flatten_negative_axis_accepted_at_opset_11() {
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", -1)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 11, &prefs).unwrap();
    }

    #[test]
    fn test_flatten_static_shape_known() {
        // Input [2, 3, 4, 5], axis=2 -> output [2*3, 4*5] = [6, 20]
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 4, Some(vec![2, 3, 4, 5]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 2)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(6), Some(20)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_flatten_static_shape_partial() {
        // Input [None, 3, 4], axis=1 -> output [None, 12]
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .add_input(
                "data",
                ArgType::Tensor(TensorType {
                    dtype: crate::ir::DType::F32,
                    rank: 3,
                    static_shape: Some(vec![None, Some(3), Some(4)]),
                }),
            )
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 1)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![None, Some(12)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_flatten_no_static_shape() {
        let mut node = create_test_node(1).build();
        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                // Even without input static_shape, output always has rank-2 shape
                assert_eq!(t.static_shape, Some(vec![None, None]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_flatten_axis_0() {
        // Input [2, 3, 4], axis=0 -> output [1, 24]
        let mut node = TestNodeBuilder::new(NodeType::Flatten, "test")
            .input_tensor_f32("data", 3, Some(vec![2, 3, 4]))
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", 0)
            .build();

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                // axis=0: left product = empty = 1, right product = 2*3*4 = 24
                assert_eq!(t.static_shape, Some(vec![Some(1), Some(24)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
