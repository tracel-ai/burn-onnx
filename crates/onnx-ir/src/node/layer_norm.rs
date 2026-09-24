//! # LayerNormalization
//!
//! Layer normalization operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LayerNormalization.html>
//!
//! ## Opset Versions
//! - **Opset 17**: Initial version introducing LayerNormalization operator. Supports `axis`,
//!   `epsilon`, and `stash_type` attributes. Includes support for optional Mean and InvStdDev outputs.
//!
//! **Implementation Note**: Requires at least 2 inputs (X and Scale; Bias is optional).
//! Accepts 1-3 outputs (Y required, optional Mean and InvStdDev). Scale and Bias are
//! lifted into a module only when Scale is a 1-D constant, `axis` is the last axis, Bias
//! is absent or constant, and neither optional output is used; otherwise both stay
//! graph values.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for LayerNorm operations
#[derive(Debug, Clone, new)]
pub struct LayerNormConfig {
    /// Small constant added for numerical stability
    pub epsilon: f64,
    /// Whether to use full precision for intermediate calculations (stash_type == 1)
    pub full_precision: bool,
    /// First normalized axis, as given in the model (negative counts from the end).
    #[new(value = "-1")]
    pub axis: i64,
}

impl LayerNormConfig {
    /// Set the epsilon value
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the full_precision value
    pub fn with_full_precision(mut self, full_precision: bool) -> Self {
        self.full_precision = full_precision;
        self
    }

    /// Set the axis value
    pub fn with_axis(mut self, axis: i64) -> Self {
        self.axis = axis;
        self
    }
}

/// Node representation for LayerNormalization operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct LayerNormalizationNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LayerNormConfig,
}

pub(crate) struct LayerNormProcessor;

/// Whether the optional Mean or InvStdDev output is used.
fn uses_statistics(node: &RawNode) -> bool {
    node.outputs.iter().skip(1).any(|arg| !arg.is_optional())
}

impl NodeProcessor for LayerNormProcessor {
    type Config = LayerNormConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 17,
            max_opset: None,
            inputs: InputSpec::AtLeast(2),
            outputs: OutputSpec::Range(1, 3),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, opset: usize) -> Result<(), ProcessError> {
        // A module holds a single [features] scale over the last axis. Anything else
        // (a multi-axis scale, or the statistics outputs) goes through the functional
        // op, which takes scale and bias as graph values.
        let rank = node.inputs[0].ty.rank() as i64;
        let axis = self.extract_config(node, opset)?.axis;
        let axis_is_last = rank > 0 && axis.rem_euclid(rank) == rank - 1;
        if node.inputs[1].ty.rank() != 1 || !axis_is_last || uses_statistics(node) {
            return Ok(());
        }
        crate::processor::lift_all_or_none(node, &[1, 2])
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        for key in node.attrs.keys() {
            match key.as_str() {
                "axis" | "epsilon" | "stash_type" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for LayerNorm: {key}"),
                    });
                }
            }
        }

        // TODO: Validate input tensor dtype is floating-point type - Type constraint T not enforced
        let input = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };

        let config = self.extract_config(node, opset)?;
        let rank = input.rank as i64;
        if config.axis < -rank || config.axis >= rank {
            return Err(ProcessError::Custom(format!(
                "LayerNorm: axis {} is out of range for rank {rank}",
                config.axis
            )));
        }
        let axis = config.axis.rem_euclid(rank) as usize;

        // Scale (and Bias) broadcast against the normalized axes X.shape[axis..].
        let normalized_rank = input.rank - axis;
        for (index, name) in [(1, "scale"), (2, "bias")] {
            if let Some(arg) = node.get_input(index)
                && arg.ty.rank() > normalized_rank
            {
                return Err(ProcessError::Custom(format!(
                    "LayerNorm: {name} must have at most rank {normalized_rank} for axis {}, got rank {}",
                    config.axis,
                    arg.ty.rank()
                )));
            }
        }

        crate::processor::same_as_input(node);

        // Mean and InvStdDev keep X's rank with the normalized axes reduced to 1. They
        // are float32 under stash_type=1, as the computation is.
        let stat_dtype = if config.full_precision {
            DType::F32
        } else {
            input.dtype
        };
        let stat_shape = input.static_shape.as_ref().map(|shape| {
            shape
                .iter()
                .enumerate()
                .map(|(i, dim)| if i < axis { *dim } else { Some(1) })
                .collect()
        });
        for output in node.outputs.iter_mut().skip(1) {
            output.ty = ArgType::Tensor(TensorType {
                dtype: stat_dtype,
                rank: input.rank,
                static_shape: stat_shape.clone(),
            });
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut epsilon = 1e-5;
        let mut stash_type = 1; // Default value is 1 (full precision)
        let mut axis = -1;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                "epsilon" => epsilon = value.clone().into_f32(),
                "stash_type" => stash_type = value.clone().into_i64(),
                _ => {}
            }
        }

        // TODO: Validate epsilon > 0 for numerical stability - Negative or zero epsilon could cause issues
        // TODO: Validate stash_type is 1 or unspecified - Spec only defines stash_type=1 (float), other values undefined
        let full_precision = stash_type == 1;
        let config = LayerNormConfig::new(epsilon as f64, full_precision).with_axis(axis);
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::LayerNormalization(LayerNormalizationNode {
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

    fn create_test_node(
        epsilon: f32,
        axis: i64,
        stash_type: i64,
        num_features: usize,
    ) -> TestNodeBuilder {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        TestNodeBuilder::new(NodeType::LayerNormalization, "test_layernorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", epsilon)
            .attr_int("axis", axis)
            .attr_int("stash_type", stash_type)
    }

    #[test]
    fn test_layer_norm_config_basic() {
        let mut node = create_test_node(1e-5, -1, 1, 64).build_with_graph_data(17);
        let processor = LayerNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 17).unwrap();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(config.full_precision); // stash_type == 1
    }

    #[test]
    fn test_layer_norm_config_no_stash_type() {
        let mut node = create_test_node(1e-5, -1, 0, 32).build_with_graph_data(17);
        let processor = LayerNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 17).unwrap();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        assert!(!config.full_precision); // stash_type == 0
    }

    #[test]
    fn test_layer_norm_rejects_scale_above_normalized_rank() {
        // axis=-1 normalizes a single axis, so a rank-2 scale cannot broadcast to it.
        let node = TestNodeBuilder::new(NodeType::LayerNormalization, "test_layernorm_invalid")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", vec![1.0; 32 * 64], vec![32, 64])
            .input_tensor_f32_data("bias", vec![0.0; 32 * 64], vec![32, 64])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", 1e-5)
            .attr_int("axis", -1)
            .attr_int("stash_type", 1)
            .build_with_graph_data(17);

        let mut node = node;
        let result = LayerNormProcessor.infer_types(&mut node, 17, &OutputPreferences::new());
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
