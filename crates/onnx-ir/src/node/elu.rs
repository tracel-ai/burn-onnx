//! # Elu Operation
//!
//! Exponential Linear Unit activation: f(x) = alpha * (exp(x) - 1) for x < 0, f(x) = x for x >= 0.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Elu.html>
//!
//! ## Type Constraints
//!
//! T: Float tensor types
//!
//! ## Opset Versions
//! - **Opset 1**: Initial support
//! - **Opset 6**: Improved shape inference
//! - **Opset 22**: Added bfloat16 support

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for Elu operation
#[derive(Debug, Clone, new)]
pub struct EluConfig {
    /// Coefficient of ELU. Default is 1.0.
    pub alpha: f64,
}

/// Node representation for Elu operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct EluNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: EluConfig,
}

/// Node processor for Elu operation
pub(crate) struct EluProcessor;

impl NodeProcessor for EluProcessor {
    type Config = EluConfig;

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
                "alpha" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Elu: {}", key),
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

        Ok(EluConfig { alpha })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Elu(EluNode {
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
        TestNodeBuilder::new(NodeType::Elu, "test_elu")
            .input_tensor_f32("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .attr_float("alpha", alpha)
            .build()
    }

    #[test]
    fn test_elu_config_with_alpha() {
        let node = create_test_node(0.5);
        let processor = EluProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert!((config.alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_elu_config_default() {
        let mut node = create_test_node(1.0);
        node.attrs.clear();
        let processor = EluProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.alpha, 1.0);
    }

    #[test]
    fn test_elu_infer_types() {
        let mut node = TestNodeBuilder::new(NodeType::Elu, "test_elu")
            .input_tensor_f32("X", 3, None)
            .output_default("Y")
            .attr_float("alpha", 1.0)
            .build();
        let processor = EluProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.dtype, crate::ir::DType::F32);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
