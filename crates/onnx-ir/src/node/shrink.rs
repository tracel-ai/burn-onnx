//! # Shrink
//!
//! Applies global average pooling to the input tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Shrink.html>
//!
//! ## Opset Versions
//! - **Opset 9**: Initial version
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for Shrink operation
#[derive(Debug, Clone, new)]
pub struct ShrinkConfig {
    /// The bias value added to output. Default is 0.
    pub bias: f64,
    /// The lambd value for the Shrink formulation. Default is 0.5.
    pub lambd: f64,
}

/// Node representation for Shrink operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ShrinkNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ShrinkConfig,
}

pub(crate) struct ShrinkProcessor;

impl NodeProcessor for ShrinkProcessor {
    type Config = ShrinkConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
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
        // Get input tensor type
        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Output has the same type and shape as input
        node.outputs[0].ty = ArgType::Tensor(input_tensor.clone());

        let _config = self.extract_config(node, _opset)?;
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut bias = 0.0;
        let mut lambd = 0.5;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "bias" => {
                    bias = value.clone().into_f32() as f64;
                }
                "lambd" => {
                    lambd = value.clone().into_f32() as f64;
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Shrink: {}", key),
                    });
                }
            }
        }

        Ok(ShrinkConfig { bias, lambd })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Shrink(ShrinkNode {
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

    fn create_test_node(bias: f32, lambd: f32) -> RawNode {
        TestNodeBuilder::new(NodeType::Shrink, "test_shrink")
            .input_tensor_f32("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .attr_float("bias", bias)
            .attr_float("lambd", lambd)
            .build()
    }

    #[test]
    fn test_shrink_config_with_bias_and_lambd() {
        let node = create_test_node(1.0, 1.5);
        let processor = ShrinkProcessor;
        let config = processor
            .extract_config(&node, 9)
            .expect("Config extraction failed");
        assert_eq!(config.bias, 1.0);
        assert_eq!(config.lambd, 1.5);
    }

    #[test]
    fn test_shrink_config_with_defaults() {
        let node = TestNodeBuilder::new(NodeType::Shrink, "test_shrink_defaults")
            .input_tensor_f32("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .build();
        let processor = ShrinkProcessor;
        let config = processor
            .extract_config(&node, 9)
            .expect("Config extraction failed");
        assert_eq!(config.bias, 0.0);
        assert_eq!(config.lambd, 0.5);
    }

    #[test]
    fn test_shrink_infer_types() {
        let mut node = TestNodeBuilder::new(NodeType::Shrink, "test_shrink_infer")
            .input_tensor_f32("X", 3, None)
            .output_default("Y")
            .attr_float("bias", 1.0)
            .attr_float("lambd", 1.5)
            .build();
        let processor = ShrinkProcessor;
        processor
            .infer_types(&mut node, 9, &OutputPreferences::default())
            .unwrap();
        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.dtype, crate::ir::DType::F32);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
