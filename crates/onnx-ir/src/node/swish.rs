//! # Swish
//!
//! Applies the Swish activation function element-wise: Swish(x) = x * sigmoid(alpha * x).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Swish.html>
//!
//! ## Attributes
//! - `alpha` (FLOAT, optional, default=1.0): Coefficient to multiply with input before sigmoid
//!
//! ## Opset Versions
//! - **Opset 24+**: Initial version

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for Swish operations
#[derive(Debug, Clone, new)]
pub struct SwishConfig {
    /// Coefficient to multiply with input before sigmoid (default: 1.0)
    pub alpha: f64,
}

/// Node representation for Swish operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SwishNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SwishConfig,
}

pub(crate) struct SwishProcessor;

impl NodeProcessor for SwishProcessor {
    type Config = SwishConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 24,
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
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "alpha" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Swish: {}", key),
                    });
                }
            }
        }

        crate::processor::same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut alpha = 1.0;
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "alpha" {
                alpha = value.clone().into_f32() as f64;
            }
        }

        Ok(SwishConfig { alpha })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Swish(SwishNode {
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

    fn create_test_node(alpha: f32) -> RawNode {
        TestNodeBuilder::new(NodeType::Swish, "test_swish")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .build()
    }

    #[test]
    fn test_swish_config_with_alpha() {
        let node = create_test_node(0.5);
        let processor = SwishProcessor;
        let config = processor.extract_config(&node, 24).unwrap();
        assert!((config.alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_swish_config_default() {
        let mut node = create_test_node(1.0);
        node.attrs.clear();
        let processor = SwishProcessor;
        let config = processor.extract_config(&node, 24).unwrap();
        assert_eq!(config.alpha, 1.0);
    }

    #[test]
    fn test_swish_infer_types() {
        let mut node = create_test_node(1.0);
        let processor = SwishProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 24, &prefs).unwrap();
        // Output should match input type
        assert_eq!(node.inputs[0].ty, node.outputs[0].ty);
    }

    #[test]
    fn test_swish_invalid_attribute() {
        let mut node = create_test_node(1.0);
        node.attrs
            .insert("bad_attr".to_string(), crate::ir::AttributeValue::Int64(1));
        let processor = SwishProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 24, &prefs);
        assert!(result.is_err());
    }
}
