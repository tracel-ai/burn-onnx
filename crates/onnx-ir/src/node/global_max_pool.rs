//! # GlobalMaxPool
//!
//! Applies global max pooling to the input tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for GlobalMaxPool operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GlobalMaxPoolNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct GlobalMaxPoolProcessor;

impl NodeProcessor for GlobalMaxPoolProcessor {
    type Config = ();

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

        // Output has the same type and rank as input (spatial dimensions become 1)
        node.outputs[0].ty = ArgType::Tensor(input_tensor.clone());

        Ok(())
    }

    fn extract_config(&self, _node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::GlobalMaxPool(GlobalMaxPoolNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_global_max_pool_type_inference() {
        let mut node = TestNodeBuilder::new(NodeType::GlobalMaxPool, "test")
            .input_tensor_f32("input", 4, None)
            .output_tensor_f32("output", 4, None)
            .build();

        let processor = GlobalMaxPoolProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 22, &prefs).unwrap();

        // Output should have same type and rank as input
        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, 4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
