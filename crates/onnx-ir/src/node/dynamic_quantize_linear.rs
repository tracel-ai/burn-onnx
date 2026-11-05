//! Dynamic per-tensor uint8 quantization (ONNX opset 11).
use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use onnx_ir_derive::NodeBuilder;
/// Quantize a float tensor while computing its scale and zero point.
#[derive(Debug, Clone, NodeBuilder)]
pub struct DynamicQuantizeLinearNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}
pub(crate) struct DynamicQuantizeLinearProcessor;
impl NodeProcessor for DynamicQuantizeLinearProcessor {
    type Config = ();
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(3),
        }
    }
    fn infer_types(
        &self,
        node: &mut RawNode,
        _: usize,
        _: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        if node.inputs[0].ty.elem_type() != DType::F32 {
            return Err(ProcessError::Custom(
                "DynamicQuantizeLinear requires float32 input".into(),
            ));
        }
        node.outputs[0].ty = match &node.inputs[0].ty {
            ArgType::Tensor(t) => {
                ArgType::Tensor(TensorType::new(DType::U8, t.rank, t.static_shape.clone()))
            }
            ArgType::ScalarTensor(_) => ArgType::ScalarTensor(DType::U8),
            _ => {
                return Err(ProcessError::Custom(
                    "DynamicQuantizeLinear requires an on-device tensor".into(),
                ));
            }
        };
        node.outputs[1].ty = ArgType::ScalarTensor(DType::F32);
        node.outputs[2].ty = ArgType::ScalarTensor(DType::U8);
        Ok(())
    }
    fn build_node(&self, node: RawNode, _: usize) -> Result<Node, ProcessError> {
        Ok(Node::DynamicQuantizeLinear(DynamicQuantizeLinearNode {
            name: node.name,
            inputs: node.inputs,
            outputs: node.outputs,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    #[test]
    fn infers_output_types() {
        let mut node = TestNodeBuilder::new(NodeType::DynamicQuantizeLinear, "dynamic")
            .input_tensor_f32("x", 2, Some(vec![2, 3]))
            .output_default("y")
            .output_default("scale")
            .output_default("zero")
            .build();
        DynamicQuantizeLinearProcessor
            .infer_types(&mut node, 11, &OutputPreferences::default())
            .unwrap();
        assert_eq!(
            node.outputs[0].ty,
            ArgType::Tensor(TensorType::new(DType::U8, 2, Some(vec![Some(2), Some(3)])))
        );
        assert_eq!(node.outputs[1].ty, ArgType::ScalarTensor(DType::F32));
        assert_eq!(node.outputs[2].ty, ArgType::ScalarTensor(DType::U8));
    }
}
