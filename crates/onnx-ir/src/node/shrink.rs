//! # Shrink
//!
//! Applies element-wise shrinkage to the input tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Shrink.html>
//!
//! ## Opset Versions
//! - **Opset 9**: Initial version
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for Shrink operation
#[derive(Debug, Clone, new)]
pub struct ShrinkConfig {
    /// The lambda value for the Shrink formulation. Default is 0.5.
    pub lambda: f64,
    /// The bias value added to output. Default is 0.
    pub bias: f64,
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
        crate::processor::same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut lambda = 0.5;
        let mut bias = 0.0;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "bias" => {
                    bias = value.clone().into_f32() as f64;
                }
                "lambd" => {
                    lambda = value.clone().into_f32() as f64;
                }
                _ => {}
            }
        }

        Ok(ShrinkConfig { lambda, bias })
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

    fn create_test_node(bias: f32, lambda: f32) -> RawNode {
        TestNodeBuilder::new(NodeType::Shrink, "test_shrink")
            .input_tensor_f32("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .attr_float("lambd", lambda)
            .attr_float("bias", bias)
            .build()
    }

    #[test]
    fn test_shrink_config_with_lambda_and_bias() {
        let node = create_test_node(1.0, 1.5);
        let processor = ShrinkProcessor;
        let config = processor
            .extract_config(&node, 9)
            .expect("Config extraction failed");
        assert_eq!(config.lambda, 1.5);
        assert_eq!(config.bias, 1.0);
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
        assert_eq!(config.lambda, 0.5);
        assert_eq!(config.bias, 0.0);
    }

    #[test]
    fn test_shrink_infer_types() {
        let mut node = TestNodeBuilder::new(NodeType::Shrink, "test_shrink_infer")
            .input_tensor_f32("X", 3, None)
            .output_default("Y")
            .attr_float("lambd", 1.5)
            .attr_float("bias", 1.0)
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
