//! # Tile
//!
//! Constructs a tensor by tiling the input tensor along specified axes.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Tile.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Used 'repeats' as an attribute (not supported in this implementation).
//! - **Opset 6**: Changed repeats from attribute to input, enabling dynamic tiling.
//! - **Opset 13**: Added support for bfloat16 and expanded type constraints.
//!
//! **Implementation Note**: This implementation requires opset 6+ (repeats as input).
//!
//! ## Example
//! Given input = [[1, 2], [3, 4]] with shape (2, 2) and repeats = [1, 2]:
//! Output = [[1, 2, 1, 2], [3, 4, 3, 4]] with shape (2, 4)
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, RuntimeInputRef, TensorDataExt};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Represents either a static value or a runtime argument for tile repeats.
#[derive(Debug, Clone)]
pub enum TileInput {
    /// Static repeats known at compile time.
    Static(Vec<usize>),
    /// Runtime repeats determined during execution.
    Runtime(RuntimeInputRef),
}

impl Default for TileInput {
    fn default() -> Self {
        TileInput::Static(vec![])
    }
}

/// Configuration for the Tile operation.
#[derive(Debug, Clone, new)]
pub struct TileConfig {
    /// The number of times to repeat each dimension.
    pub repeats: TileInput,
}

/// Node representation for Tile operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct TileNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: TileConfig,
}

pub(crate) struct TileProcessor;

impl NodeProcessor for TileProcessor {
    type Config = TileConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift repeats input (input[1]) if present
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate repeats when statically known
        if let Some(repeats_arg) = node.inputs.get(1)
            && let Some(tensor_data) = repeats_arg.value()
        {
            let i64_values: Vec<i64> =
                tensor_data
                    .to_i64_vec()
                    .map_err(|_| ProcessError::TypeMismatch {
                        expected: "Int64".to_string(),
                        actual: format!("{:?}", tensor_data.elem_type()),
                    })?;

            // Validate non-negative
            if let Some(neg) = i64_values.iter().find(|&&v| v < 0) {
                return Err(ProcessError::Custom(format!(
                    "Tile: repeats values must be non-negative, got {}",
                    neg
                )));
            }

            // Validate length matches input rank
            let input_rank = match &node.inputs[0].ty {
                ArgType::Tensor(t) => Some(t.rank),
                _ => None,
            };
            if let Some(rank) = input_rank
                && i64_values.len() != rank
            {
                return Err(ProcessError::Custom(format!(
                    "Tile: repeats length ({}) must match input rank ({})",
                    i64_values.len(),
                    rank
                )));
            }
        }

        // Infer output type - same as input
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract repeats config (always an input, all opset versions)
        fn get_repeats(node: &RawNode) -> TileInput {
            if let Some(input) = node.inputs.get(1) {
                match input.value() {
                    None => {
                        // Runtime input - store reference instead of cloning the argument
                        TileInput::Runtime(RuntimeInputRef::new(input.name.clone(), 1))
                    }
                    Some(tensor_data) => {
                        let i64_values: Vec<i64> = tensor_data.try_into_vec().unwrap();
                        let repeats = i64_values.iter().map(|&x| x as usize).collect();
                        TileInput::Static(repeats)
                    }
                }
            } else {
                // No repeats input provided - default to empty
                TileInput::Static(vec![])
            }
        }

        let repeats = get_repeats(node);
        let config = TileConfig { repeats };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Tile(TileNode {
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

    /// Helper function to create test nodes with different repeat values
    fn create_test_node(repeats: Option<Vec<i64>>, input_rank: usize) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Tile, "test_tile")
            .input_tensor_f32("input", input_rank, None)
            .output_tensor_f32("output", input_rank, None); // Same rank as input initially

        // Add repeats input if provided
        if let Some(reps) = repeats {
            builder = builder.input_tensor_i64_data("repeats", reps.clone(), vec![reps.len()]);
        }

        builder
    }

    #[test]
    fn test_tile_config_with_repeats() {
        // Test with normal repeats values
        let repeats = vec![2, 3, 4];
        let node = create_test_node(Some(repeats.clone()), 3).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should extract repeats correctly
        assert!(matches!(&config.repeats, TileInput::Static(r) if r == &vec![2, 3, 4]));
    }

    #[test]
    fn test_tile_config_with_single_repeat() {
        // Test with single repeat value
        let repeats = vec![5];
        let node = create_test_node(Some(repeats.clone()), 1).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(&config.repeats, TileInput::Static(r) if r == &vec![5]));
    }

    #[test]
    fn test_tile_config_with_zero_repeats() {
        // Test with repeats including zeros
        let repeats = vec![0, 1, 0];
        let node = create_test_node(Some(repeats.clone()), 3).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(&config.repeats, TileInput::Static(r) if r == &vec![0, 1, 0]));
    }

    #[test]
    fn test_tile_config_with_large_repeats() {
        // Test with large repeats values
        let repeats = vec![100, 200];
        let node = create_test_node(Some(repeats.clone()), 2).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(&config.repeats, TileInput::Static(r) if r == &vec![100, 200]));
    }

    #[test]
    fn test_tile_config_without_repeats_input() {
        // Test when repeats input is missing
        let node = create_test_node(None, 3).build();

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should return empty repeats
        assert!(matches!(&config.repeats, TileInput::Static(r) if r.is_empty()));
    }

    #[test]
    fn test_tile_config_with_negative_repeats_rejected() {
        let repeats = vec![-1, 2, -3];
        let mut node = create_test_node(Some(repeats), 3).build_with_graph_data(16);

        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("non-negative"))
        );
    }

    #[test]
    fn test_tile_config_with_empty_repeats_rejected() {
        // Empty repeats (length 0) doesn't match input rank 3
        let repeats = vec![];
        let mut node = create_test_node(Some(repeats), 3).build_with_graph_data(16);

        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("must match input rank"))
        );
    }

    #[test]
    fn test_tile_config_with_runtime_repeats() {
        // Test with repeats input that has no static value (runtime)
        let mut node = create_test_node(None, 3).build();

        // Add repeats input with no value
        node.inputs.push(
            TestNodeBuilder::new(NodeType::Identity, "temp")
                .input_tensor_i64("repeats", 1, Some(vec![3]))
                .build()
                .inputs
                .pop()
                .unwrap(),
        );

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should return Runtime repeats
        assert!(matches!(&config.repeats, TileInput::Runtime(arg) if arg.name == "repeats"));
    }

    #[test]
    fn test_tile_repeats_length_mismatch_rejected() {
        // input rank=3 but repeats has 2 elements
        let repeats = vec![2, 3];
        let mut node = create_test_node(Some(repeats), 3).build_with_graph_data(16);

        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("must match input rank"))
        );
    }
}
