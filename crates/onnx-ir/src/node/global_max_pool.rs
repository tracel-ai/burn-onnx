//! # GlobalMaxPool
//!
//! Applies global max pooling to the input tensor: the maximum over every spatial axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
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
        opset: usize,
        output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Same input checks and output shape as GlobalAveragePool.
        crate::node::global_avg_pool::GlobalAveragePoolProcessor.infer_types(
            node,
            opset,
            output_preferences,
        )
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
    use crate::ir::{ArgType, DType, NodeType, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_global_max_pool_output_collapses_spatial_axes() {
        let mut node = TestNodeBuilder::new(NodeType::GlobalMaxPool, "test_global_max_pool")
            .input_tensor_f32("X", 4, Some(vec![2, 3, 5, 7]))
            .output_default("Y")
            .build();
        GlobalMaxPoolProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        assert_eq!(
            node.outputs[0].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 4,
                static_shape: Some(vec![Some(2), Some(3), Some(1), Some(1)]),
            })
        );
    }

    #[test]
    fn test_global_max_pool_rejects_rank_2() {
        let mut node = TestNodeBuilder::new(NodeType::GlobalMaxPool, "test_global_max_pool")
            .input_tensor_f32("X", 2, None)
            .output_default("Y")
            .build();
        let result = GlobalMaxPoolProcessor.infer_types(&mut node, 16, &OutputPreferences::new());
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
