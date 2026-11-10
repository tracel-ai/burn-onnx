//! # Bernoulli
//!
//! Draws binary random numbers (0 or 1) from a Bernoulli distribution.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Bernoulli.html>
//!
//! ## Opset Versions
//!
//! - **Opset 15**: Initial version with dtype and seed attributes for drawing binary random numbers
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, BoolStore, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

/// Attributes for Bernoulli sampling.
#[derive(Debug, Clone, Default)]
pub struct BernoulliConfig {
    /// Optional output dtype code; omission preserves the input dtype.
    pub dtype: Option<i64>,
    /// Optional ONNX random seed.
    pub seed: Option<f32>,
}

/// Node representation for Bernoulli operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct BernoulliNode {
    pub config: BernoulliConfig,
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct BernoulliProcessor;

impl NodeProcessor for BernoulliProcessor {
    type Config = BernoulliConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 15,
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
        // TODO: Add validation for unexpected attributes

        // Get the tensor type and its rank
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };
        let rank = tensor.rank;
        let static_shape = tensor.static_shape.clone();

        // Infer elem type based on the dtype attribute
        let dtype = node
            .attrs
            .get("dtype")
            .map(|val| {
                let code = val.clone().into_i32()?;
                DataType::from_i32(code).ok_or_else(|| {
                    ProcessError::Custom(format!("unknown dtype attribute value {code}"))
                })
            })
            .transpose()?;

        let elem_type = dtype.map_or(tensor.dtype, |dtype| match dtype {
            DataType::FLOAT => DType::F32,
            DataType::INT32 => DType::I32,
            DataType::INT64 => DType::I64,
            DataType::DOUBLE => DType::F64,
            DataType::BOOL => DType::Bool(BoolStore::Native),
            _ => tensor.dtype, // Fallback to input type for unsupported dtype
        });

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: elem_type,
            rank,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(BernoulliConfig {
            dtype: node
                .attrs
                .get("dtype")
                .map(|v| v.clone().into_i64())
                .transpose()?,
            seed: node
                .attrs
                .get("seed")
                .map(|v| v.clone().into_f32())
                .transpose()?,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;
        Ok(Node::Bernoulli(BernoulliNode {
            config,
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: Option<i32>, static_shape: Option<Vec<usize>>) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::Bernoulli, "test_bernoulli")
            .input_tensor_f32("input", 4, static_shape) // Rank 0 will be updated
            .output_tensor_f32("output", 0, None); // Rank 0 will be updated

        if let Some(dtype) = dtype {
            builder = builder.attr_int("dtype", dtype as i64)
        }

        builder.build()
    }

    #[test]
    fn preserves_seed_and_dtype() {
        let node = TestNodeBuilder::new(NodeType::Bernoulli, "seeded")
            .attr_float("seed", 42.5)
            .attr_int("dtype", 7)
            .build();
        let config = BernoulliProcessor.extract_config(&node, 22).unwrap();
        assert_eq!(config.seed, Some(42.5));
        assert_eq!(config.dtype, Some(7));
        let node = TestNodeBuilder::new(NodeType::Bernoulli, "unseeded").build();
        assert_eq!(
            BernoulliProcessor.extract_config(&node, 22).unwrap().seed,
            None
        );
    }

    #[test]
    fn test_bernoulli_int() {
        let mut node = create_test_node(Some(DataType::INT32.value()), Some(vec![3, 4, 2]));
        let processor = BernoulliProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I32);
                assert_eq!(tensor.static_shape, Some(vec![Some(3), Some(4), Some(2)]));
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_bernoulli_no_cast() {
        let mut node = create_test_node(None, Some(vec![3, 4, 2]));
        let processor = BernoulliProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_bernoulli_no_static_shape() {
        let mut node = create_test_node(None, None);
        let processor = BernoulliProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_bernoulli_invalid_input() {
        let mut node = create_test_node(Some(DataType::FLOAT.value()), None);
        node.inputs[0].ty = ArgType::ScalarNative(DType::F32);
        let processor = BernoulliProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
}
