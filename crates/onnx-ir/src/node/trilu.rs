//! # Trilu
//!
//! Returns the upper or lower triangular part of a 2-D matrix or batches of 2-D matrices.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Trilu.html>
//!
//! ## Opset Versions
//! - **Opset 14**: Initial version introducing triangular matrix extraction with optional diagonal offset.
//!
//! ## Implementation Notes
//! - If `upper=1` (true):
//!   - Positive k: Retains upper triangle excluding main diagonal and (k-1) diagonals above it
//!   - Negative k: Retains upper triangle including main diagonal and |k| diagonals below it
//! - If `upper=0` (false):
//!   - Positive k: Retains lower triangle including main diagonal and k diagonals above it
//!   - Negative k: Retains lower triangle excluding main diagonal and (|k|-1) diagonals below it

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, RuntimeInputRef, TensorDataExt};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// The diagonal offset `k`, known at build time or read from an input at run time.
#[derive(Debug, Clone, PartialEq)]
pub enum TriluDiagonal {
    /// Offset known at build time (0 when the input is absent).
    Static(i64),
    /// Offset read from a runtime scalar input.
    Runtime(RuntimeInputRef),
}

/// Configuration for the Trilu operation.
#[derive(Debug, Clone, PartialEq, new)]
pub struct TriluConfig {
    /// Whether to return the upper triangular matrix.
    pub upper: bool,
    /// The diagonal offset.
    pub diagonal: TriluDiagonal,
}

/// Node representation for Trilu operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct TriluNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: TriluConfig,
}

pub(crate) struct TriluProcessor;

impl NodeProcessor for TriluProcessor {
    type Config = TriluConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 14,
            max_opset: None,
            inputs: InputSpec::Range(1, 2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift diagonal input (input[1]) if present, not optional, and constant
        if node.inputs.len() > 1 && !node.inputs[1].is_optional() && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let input_rank = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.rank,
            _ => 0,
        };

        if input_rank < 2 {
            return Err(ProcessError::Custom(format!(
                "Trilu: input must have rank >= 2, got rank {}",
                input_rank
            )));
        }

        self.extract_config(node, opset)?;

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut upper = true;
        let mut diagonal = TriluDiagonal::Static(0);
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "upper" {
                upper = value.clone().into_i64() != 0
            }
        }
        if let Some(diagonal_arg) = node.get_input(1) {
            diagonal = match diagonal_arg.value() {
                Some(tensor_data) => {
                    TriluDiagonal::Static(tensor_data.scalar_i64().map_err(|e| {
                        ProcessError::Custom(format!(
                            "Trilu node {}: failed to read diagonal value: {e:?}",
                            node.name
                        ))
                    })?)
                }
                // A runtime k is passed to burn's tril/triu, which take a native scalar.
                None if diagonal_arg.ty.is_scalar() => {
                    TriluDiagonal::Runtime(RuntimeInputRef::new(diagonal_arg.name.clone(), 1))
                }
                None => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "scalar diagonal offset k".to_string(),
                        actual: format!("{:?}", diagonal_arg.ty),
                    });
                }
            };
        }

        let config = TriluConfig::new(upper, diagonal);
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Trilu(TriluNode {
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

    /// Helper function to create test nodes for Trilu tests
    fn create_test_node(upper_attr: Option<i64>, diagonal_input: Option<i64>) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Trilu, "test_trilu")
            .input_tensor_f32("X", 2, None) // Typically a matrix
            .output_tensor_f32("Y", 2, None);

        // Add diagonal input if provided
        if let Some(diag) = diagonal_input {
            builder = builder.input_scalar_tensor_i64("k", Some(diag));
        }

        // Add upper attribute if provided
        if let Some(upper) = upper_attr {
            builder = builder.attr_int("upper", upper);
        }

        builder
    }

    #[test]
    fn test_trilu_config_default() {
        // Test with no attributes or inputs - should use defaults (upper=true, diagonal=0)
        let node = create_test_node(None, None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: TriluDiagonal::Static(0)
            }
        );
    }

    #[test]
    fn test_trilu_config_upper_true() {
        // Test with upper=1 attribute
        let node = create_test_node(Some(1), None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: TriluDiagonal::Static(0)
            }
        );
    }

    #[test]
    fn test_trilu_config_upper_false() {
        // Test with upper=0 attribute (lower triangular)
        let node = create_test_node(Some(0), None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(
            config,
            TriluConfig {
                upper: false,
                diagonal: TriluDiagonal::Static(0)
            }
        );
    }

    #[test]
    fn test_trilu_config_with_diagonal() {
        // Test with diagonal=2 input (offset 2 above main diagonal)
        let node = create_test_node(None, Some(2)).build_with_graph_data(16);

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: TriluDiagonal::Static(2)
            }
        );
    }

    #[test]
    fn test_trilu_config_with_negative_diagonal() {
        // Test with diagonal=-3 input (offset 3 below main diagonal)
        let node = create_test_node(None, Some(-3)).build_with_graph_data(16);

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: TriluDiagonal::Static(-3)
            }
        );
    }

    #[test]
    fn test_trilu_config_both_params() {
        // Test with both upper attribute and diagonal input
        let node = create_test_node(Some(0), Some(1)).build_with_graph_data(16);

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(
            config,
            TriluConfig {
                upper: false,
                diagonal: TriluDiagonal::Static(1)
            }
        );
    }

    #[test]
    fn test_trilu_config_runtime_diagonal() {
        // A diagonal fed by a graph input is read at run time, not treated as 0.
        let node = TestNodeBuilder::new(NodeType::Trilu, "test_trilu")
            .input_tensor_f32("X", 2, None)
            .input_scalar_i64("k")
            .output_tensor_f32("Y", 2, None)
            .attr_int("upper", 0)
            .build();

        let config = TriluProcessor.extract_config(&node, 16).unwrap();
        assert_eq!(
            config.diagonal,
            TriluDiagonal::Runtime(RuntimeInputRef::new("k".to_string(), 1))
        );
    }

    #[test]
    fn test_trilu_rejects_non_scalar_runtime_diagonal() {
        let node = TestNodeBuilder::new(NodeType::Trilu, "test_trilu")
            .input_tensor_f32("X", 2, None)
            .input_tensor_i64("k", 1, None)
            .output_tensor_f32("Y", 2, None)
            .build();

        let result = TriluProcessor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_trilu_config_non_binary_upper() {
        // Test with non-binary values for the upper attribute
        // Any non-zero value should be treated as true
        let node = create_test_node(Some(42), None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: TriluDiagonal::Static(0)
            }
        );
    }

    #[test]
    fn test_trilu_config_negative_non_binary_upper() {
        // Test with negative values for the upper attribute
        // Any non-zero value should be treated as true
        let node = create_test_node(Some(-5), None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: TriluDiagonal::Static(0)
            }
        );
    }

    #[test]
    fn test_trilu_rank_1_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::Trilu, "test_trilu")
            .input_tensor_f32("X", 1, None) // 1D input
            .output_tensor_f32("Y", 1, None)
            .build();

        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("rank >= 2")));
    }
}
