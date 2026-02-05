//! # BatchNormalization
//!
//! Batch normalization operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__BatchNormalization.html>
//!
//! ## Opset Versions
//! - **Opset 1-5**: Initial version with spatial attribute
//! - **Opset 6-8**: Removed spatial attribute, added consumed_inputs
//! - **Opset 9-13**: Removed consumed_inputs attribute
//! - **Opset 14-15**: Added training_mode attribute, expanded type support
//! - **Opset 15+**: Current version with full training mode support
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for BatchNorm operations.
///
/// When all weight inputs (scale, bias, mean, var) are static initializers,
/// we use `Static` which allows generating a `BatchNorm` module.
/// When any weight input comes from another node at runtime, we use `Runtime`
/// which generates inline math in the forward pass.
#[derive(Debug, Clone)]
pub enum BatchNormConfig {
    /// All weights are static initializers → use BatchNorm module
    Static(BatchNormStaticConfig),
    /// Some weights are runtime tensors → generate inline math
    Runtime(BatchNormRuntimeConfig),
}

/// Static BatchNorm config — all weights are known at build time.
#[derive(Debug, Clone, new)]
pub struct BatchNormStaticConfig {
    /// Small constant added for numerical stability
    pub epsilon: f64,
    /// Momentum for running statistics
    pub momentum: f64,
}

/// Runtime BatchNorm config — weights come from other nodes.
#[derive(Debug, Clone, new)]
pub struct BatchNormRuntimeConfig {
    /// Small constant added for numerical stability
    pub epsilon: f64,
    /// Momentum for running statistics
    pub momentum: f64,
}

/// Node representation for BatchNormalization operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct BatchNormalizationNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: BatchNormConfig,
}

pub(crate) struct BatchNormProcessor;

impl NodeProcessor for BatchNormProcessor {
    type Config = BatchNormConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
            max_opset: None,
            inputs: InputSpec::Exact(5),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Only lift weight inputs to static if ALL four are constants.
        // Partially lifting would break: the Static path needs all 4 as snapshots,
        // and the Runtime path needs all 4 as forward inputs.
        let all_constant = (1..=4).all(|i| node.inputs.len() > i && node.inputs[i].is_constant());

        if all_constant {
            for i in 1..=4 {
                node.inputs[i].to_static()?;
            }
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add validation for unexpected attributes
        // FIXME: Check training_mode attribute - spec mentions it but implementation doesn't validate it
        // According to spec, training mode outputs mean/var/saved_mean/saved_var which are not currently handled
        // TODO: Add test coverage for training_mode=1 case - spec says outputs 5 tensors but only 1 output validated
        // TODO: Validate epsilon and momentum ranges - negative epsilon would be invalid, momentum should be [0,1]
        // TODO: Add test for mismatched input tensor shapes - scale/bias/mean/var must match channels dimension
        // TODO: Add test for wrong input tensor ranks - spec requires scale/bias/mean/var to be 1D
        // TODO: Validate input[3] and input[4] are actually mean and variance tensors (rank 1)

        // Extract input tensor type
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // BatchNorm preserves rank (same as input)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut epsilon = 0f32;
        let mut momentum = 0f32;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "momentum" => momentum = value.clone().into_f32(),
                "epsilon" => epsilon = value.clone().into_f32(),
                _ => {}
            }
        }

        // Check if all weight inputs (1-4) have static values
        let all_static = (1..=4).all(|i| node.inputs[i].value().is_some());

        if all_static {
            Ok(BatchNormConfig::Static(BatchNormStaticConfig::new(
                epsilon as f64,
                momentum as f64,
            )))
        } else {
            Ok(BatchNormConfig::Runtime(BatchNormRuntimeConfig::new(
                epsilon as f64,
                momentum as f64,
            )))
        }
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::BatchNormalization(BatchNormalizationNode {
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

    fn create_test_node(epsilon: f32, momentum: f32, num_features: usize) -> TestNodeBuilder {
        let ones = vec![1.0; num_features];
        let zeros = vec![0.0; num_features];

        TestNodeBuilder::new(NodeType::BatchNormalization, "test_batchnorm")
            .input_tensor_f32("X", 4, None) // NCHW format
            .input_tensor_f32_data("scale", ones.clone(), vec![num_features])
            .input_tensor_f32_data("bias", zeros.clone(), vec![num_features])
            .input_tensor_f32_data("mean", zeros.clone(), vec![num_features])
            .input_tensor_f32_data("var", ones.clone(), vec![num_features])
            .output_tensor_f32("output", 4, None)
            .attr_float("epsilon", epsilon)
            .attr_float("momentum", momentum)
    }

    fn create_runtime_test_node(epsilon: f32, momentum: f32) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::BatchNormalization, "test_batchnorm")
            .input_tensor_f32("X", 4, None)
            .input_tensor_f32("scale", 1, None)
            .input_tensor_f32("bias", 1, None)
            .input_tensor_f32("mean", 1, None)
            .input_tensor_f32("var", 1, None)
            .output_tensor_f32("output", 4, None)
            .attr_float("epsilon", epsilon)
            .attr_float("momentum", momentum)
    }

    #[test]
    fn test_batch_norm_config_basic() {
        let node = create_test_node(1e-5, 0.9, 64).build_with_graph_data(16);
        let mut node = node;
        let processor = BatchNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match config {
            BatchNormConfig::Static(c) => {
                assert!(f64::abs(c.epsilon - 1e-5) < 1e-6);
                assert!(f64::abs(c.momentum - 0.9) < 1e-6);
            }
            _ => panic!("Expected Static config"),
        }
    }

    #[test]
    fn test_batch_norm_config_default_values() {
        let node = create_test_node(0.0, 0.0, 32).build_with_graph_data(16);
        let mut node = node;
        let processor = BatchNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match config {
            BatchNormConfig::Static(c) => {
                assert!(f64::abs(c.epsilon - 0.0) < 1e-6);
                assert!(f64::abs(c.momentum - 0.0) < 1e-6);
            }
            _ => panic!("Expected Static config"),
        }
    }

    #[test]
    fn test_batch_norm_config_runtime() {
        let node = create_runtime_test_node(1e-5, 0.9).build_with_graph_data(16);
        let processor = BatchNormProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        match config {
            BatchNormConfig::Runtime(c) => {
                assert!(f64::abs(c.epsilon - 1e-5) < 1e-6);
                assert!(f64::abs(c.momentum - 0.9) < 1e-6);
            }
            _ => panic!("Expected Runtime config"),
        }
    }
}
