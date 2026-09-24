//! # GlobalAveragePool
//!
//! Applies global average pooling to the input tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for GlobalAveragePool operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GlobalAveragePoolNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct GlobalAveragePoolProcessor;

impl NodeProcessor for GlobalAveragePoolProcessor {
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
        // Matches ORT, which reports "Input dimension cannot be less than 3".
        if input_tensor.rank <= 2 {
            return Err(ProcessError::Custom(format!(
                "input tensor requires rank at least 3, got rank {}",
                input_tensor.rank
            )));
        }

        node.outputs[0].ty = ArgType::Tensor(global_pool_output_type(input_tensor));

        Ok(())
    }

    fn extract_config(&self, _node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::GlobalAveragePool(GlobalAveragePoolNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

/// Output type of a global pool: same dtype and rank as the input, N and C carried
/// through, every spatial dim collapsed to 1. Shared with GlobalLpPool.
pub(crate) fn global_pool_output_type(input: &TensorType) -> TensorType {
    // Length comes from `rank`, so it cannot drift from the rank written below
    // if the input's `static_shape` disagrees with its own rank.
    let mut static_shape = vec![None; input.rank];
    if let Some(input_shape) = &input.static_shape {
        for (out, inp) in static_shape.iter_mut().zip(input_shape).take(2) {
            *out = *inp;
        }
    }
    for el in static_shape.iter_mut().skip(2) {
        *el = Some(1usize);
    }

    TensorType {
        dtype: input.dtype,
        rank: input.rank,
        static_shape: Some(static_shape),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_global_avg_pool_type_inference() {
        let mut node = TestNodeBuilder::new(NodeType::GlobalAveragePool, "test")
            .input_tensor_f32("input", 4, None)
            .output_tensor_f32("output", 4, None)
            .build();

        let processor = GlobalAveragePoolProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should have same type and rank as input
        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, 4);
        } else {
            panic!("Expected Tensor output");
        }
    }

    /// N and C carry through, every spatial dim collapses to 1.
    #[test]
    fn test_global_avg_pool_output_static_shape() {
        let cases = [
            (3, None, vec![None, None, Some(1)]),
            (4, None, vec![None, None, Some(1), Some(1)]),
            (
                4,
                Some(vec![1, 3, 8, 8]),
                vec![Some(1), Some(3), Some(1), Some(1)],
            ),
            (
                5,
                Some(vec![2, 4, 6, 7, 8]),
                vec![Some(2), Some(4), Some(1), Some(1), Some(1)],
            ),
        ];
        for (rank, input_shape, expected) in cases {
            let mut node = TestNodeBuilder::new(NodeType::GlobalAveragePool, "test")
                .input_tensor_f32("input", rank, input_shape.clone())
                .output_tensor_f32("output", rank, None)
                .build();
            GlobalAveragePoolProcessor
                .infer_types(&mut node, 16, &OutputPreferences::new())
                .unwrap();
            let ArgType::Tensor(output_tensor) = &node.outputs[0].ty else {
                panic!("Expected Tensor output");
            };
            assert_eq!(output_tensor.rank, rank);
            assert_eq!(
                output_tensor.static_shape,
                Some(expected),
                "rank {rank}, input static_shape {input_shape:?}"
            );
        }
    }

    /// A partially known input shape keeps its unknown N as `None` and still
    /// collapses the spatial dims.
    #[test]
    fn test_global_avg_pool_partial_static_shape() {
        let mut node = TestNodeBuilder::new(NodeType::GlobalAveragePool, "test")
            .input_tensor_f32("input", 4, None)
            .output_tensor_f32("output", 4, None)
            .build();
        let ArgType::Tensor(input_ty) = &mut node.inputs[0].ty else {
            panic!("Expected Tensor input");
        };
        input_ty.static_shape = Some(vec![None, Some(3), Some(8), Some(8)]);

        GlobalAveragePoolProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        let ArgType::Tensor(output_tensor) = &node.outputs[0].ty else {
            panic!("Expected Tensor output");
        };
        assert_eq!(
            output_tensor.static_shape,
            Some(vec![None, Some(3), Some(1), Some(1)])
        );
    }

    /// The spec requires N x C x D1 ... Dn, so rank 2 and below have no spatial dims.
    #[test]
    fn test_global_avg_pool_rejects_rank_below_3() {
        for rank in [1, 2] {
            let mut node = TestNodeBuilder::new(NodeType::GlobalAveragePool, "test")
                .input_tensor_f32("input", rank, None)
                .output_tensor_f32("output", rank, None)
                .build();
            let result =
                GlobalAveragePoolProcessor.infer_types(&mut node, 16, &OutputPreferences::new());
            assert!(
                matches!(result, Err(ProcessError::Custom(_))),
                "rank {rank} should be rejected, got {result:?}"
            );
        }
    }
}
