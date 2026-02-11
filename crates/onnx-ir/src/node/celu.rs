//! # Celu Operation
//!
//! Continuously Differentiable Exponential Linear Units:
//! y = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Celu.html>
//!
//! ## Type Constraints
//!
//! T: tensor(float) - constrained to float32 tensors
//!
//! ## Opset Versions
//! - **Opset 12**: Initial support

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for Celu operation
#[derive(Debug, Clone, new)]
pub struct CeluConfig {
    /// The alpha value that controls the shape of the unit. Default is 1.0.
    pub alpha: f64,
}

/// Node representation for Celu operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct CeluNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: CeluConfig,
}

/// Node processor for Celu operation
pub(crate) struct CeluProcessor;

impl NodeProcessor for CeluProcessor {
    type Config = CeluConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 12,
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
                        reason: format!("Unexpected attribute for Celu: {}", key),
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

        Ok(CeluConfig { alpha })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Celu(CeluNode {
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
        TestNodeBuilder::new(NodeType::Celu, "test_celu")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .build()
    }

    #[test]
    fn test_celu_config_with_alpha() {
        let node = create_test_node(2.0);
        let processor = CeluProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert!((config.alpha - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_celu_config_default() {
        let mut node = create_test_node(1.0);
        node.attrs.clear();
        let processor = CeluProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.alpha, 1.0);
    }

    #[test]
    fn test_celu_infer_types() {
        let mut node = TestNodeBuilder::new(NodeType::Celu, "test_celu")
            .input_tensor_f32("X", 4, None)
            .output_default("Y")
            .attr_float("alpha", 1.0)
            .build();
        let processor = CeluProcessor;
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
