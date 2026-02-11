//! # Selu Operation
//!
//! Scaled Exponential Linear Unit activation:
//! y = gamma * (alpha * e^x - alpha) for x <= 0, y = gamma * x for x > 0
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Selu.html>
//!
//! ## Attributes
//! - alpha (float, default 1.67326319217681884765625)
//! - gamma (float, default 1.05070102214813232421875)
//!
//! ## Type Constraints
//!
//! T: Float tensor types
//!
//! ## Opset Versions
//! - **Opset 1**: Initial support
//! - **Opset 6**: Updated type support
//! - **Opset 22**: Added bfloat16 support

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for Selu operation
#[derive(Debug, Clone, new)]
pub struct SeluConfig {
    /// Alpha coefficient. Default is 1.67326319217681884765625.
    pub alpha: f64,
    /// Gamma coefficient. Default is 1.05070102214813232421875.
    pub gamma: f64,
}

/// Node representation for Selu operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SeluNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SeluConfig,
}

/// Node processor for Selu operation
pub(crate) struct SeluProcessor;

impl NodeProcessor for SeluProcessor {
    type Config = SeluConfig;

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
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "alpha" | "gamma" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Selu: {}", key),
                    });
                }
            }
        }

        same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut alpha = 1.673_263_192_176_818_8_f64;
        let mut gamma = 1.050_701_022_148_132_3_f64;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "alpha" => alpha = value.clone().into_f32() as f64,
                "gamma" => gamma = value.clone().into_f32() as f64,
                _ => {}
            }
        }

        Ok(SeluConfig { alpha, gamma })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Selu(SeluNode {
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

    fn create_test_node() -> RawNode {
        TestNodeBuilder::new(NodeType::Selu, "test_selu")
            .input_tensor_f32("X", 4, None)
            .output_default("Y")
            .build()
    }

    #[test]
    fn test_selu_config_default() {
        let node = create_test_node();
        let processor = SeluProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert!((config.alpha - 1.67326319217681884765625).abs() < 1e-6);
        assert!((config.gamma - 1.05070102214813232421875).abs() < 1e-6);
    }

    #[test]
    fn test_selu_config_custom() {
        let mut node = create_test_node();
        node.attrs
            .insert("alpha".to_string(), crate::ir::AttributeValue::Float32(2.0));
        node.attrs
            .insert("gamma".to_string(), crate::ir::AttributeValue::Float32(0.5));
        let processor = SeluProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert!((config.alpha - 2.0).abs() < 1e-6);
        assert!((config.gamma - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_selu_infer_types() {
        let mut node = create_test_node();
        let processor = SeluProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                assert_eq!(t.dtype, crate::ir::DType::F32);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
