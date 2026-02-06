//! # GatherND
//!
//! Gathers slices from data into an output tensor using multi-dimensional index tuples.
//! This is the inverse of ScatterND.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GatherND.html>
//!
//! ## Opset Versions
//! - **Opset 11**: Initial version.
//! - **Opset 12**: Added batch_dims attribute, bfloat16 support.
//! - **Opset 13**: No functional changes.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for the GatherND operation.
#[derive(Debug, Clone, new, Default)]
pub struct GatherNDConfig {
    /// Number of leading batch dimensions (default: 0).
    pub batch_dims: usize,
}

/// Node representation for GatherND operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct GatherNDNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: GatherNDConfig,
}

pub(crate) struct GatherNDProcessor;

impl NodeProcessor for GatherNDProcessor {
    type Config = GatherNDConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let data_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.clone(),
            other => {
                return Err(ProcessError::Custom(format!(
                    "GatherND data input must be a tensor, got {:?}",
                    other
                )));
            }
        };

        let indices_tensor = match &node.inputs[1].ty {
            ArgType::Tensor(t) => t.clone(),
            other => {
                return Err(ProcessError::Custom(format!(
                    "GatherND indices input must be a tensor, got {:?}",
                    other
                )));
            }
        };

        let r = data_tensor.rank;
        let q = indices_tensor.rank;

        // Extract batch_dims attribute (default: 0)
        let mut batch_dims: i64 = 0;
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "batch_dims" => batch_dims = value.clone().into_i64(),
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for GatherND: {}", key),
                    });
                }
            }
        }

        let b = batch_dims as usize;

        // Validate: b < min(q, r)
        if b >= q.min(r) {
            return Err(ProcessError::InvalidAttribute {
                name: "batch_dims".to_string(),
                reason: format!(
                    "batch_dims {} must be < min(indices_rank={}, data_rank={})",
                    b, q, r
                ),
            });
        }

        // Output rank = q + r - indices_shape[-1] - 1 - b
        // We don't know indices_shape[-1] statically, but we know it from
        // the static shape if available. If not available, we cannot infer
        // the output rank, which is a problem.
        //
        // For now, require that indices has a static shape so we can
        // determine the last dimension (k).
        let k = indices_tensor
            .static_shape
            .as_ref()
            .and_then(|s| s.last().copied())
            .flatten()
            .ok_or_else(|| {
                ProcessError::Custom(
                    "GatherND requires indices to have a known last dimension to determine output rank"
                        .to_string(),
                )
            })?;

        // Validate: k <= r - b
        if k > r - b {
            return Err(ProcessError::Custom(format!(
                "GatherND indices last dimension {} must be <= data_rank({}) - batch_dims({})",
                k, r, b
            )));
        }

        // output_rank = q + r - k - 1 - b
        let output_rank = q + r - k - 1 - b;

        if output_rank == 0 {
            node.outputs[0].ty = ArgType::Scalar(data_tensor.dtype);
        } else {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: data_tensor.dtype,
                rank: output_rank,
                static_shape: None,
            });
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut batch_dims: i64 = 0;
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "batch_dims" => batch_dims = value.clone().into_i64(),
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for GatherND: {}", key),
                    });
                }
            }
        }

        Ok(GatherNDConfig {
            batch_dims: batch_dims as usize,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::GatherND(GatherNDNode {
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
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(
        data_rank: usize,
        indices_rank: usize,
        indices_last_dim: usize,
        batch_dims: i64,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::GatherND, "test_gathernd");

        if batch_dims != 0 {
            builder = builder.attr_int("batch_dims", batch_dims);
        }

        // Build a static shape for indices: [2; indices_rank] with last dim = indices_last_dim
        let mut indices_shape = vec![2usize; indices_rank];
        *indices_shape.last_mut().unwrap() = indices_last_dim;

        builder
            .input_tensor_f32("data", data_rank, None)
            .input_tensor_i64("indices", indices_rank, Some(indices_shape))
            .output_tensor_f32("output", 1, None) // rank will be updated by type inference
    }

    #[test]
    fn test_config_default_batch_dims() {
        let node = create_test_node(2, 2, 2, 0).build();
        let processor = GatherNDProcessor;
        let config = processor.extract_config(&node, 12).unwrap();
        assert_eq!(config.batch_dims, 0);
    }

    #[test]
    fn test_config_custom_batch_dims() {
        let node = create_test_node(3, 2, 1, 1).build();
        let processor = GatherNDProcessor;
        let config = processor.extract_config(&node, 12).unwrap();
        assert_eq!(config.batch_dims, 1);
    }

    #[test]
    fn test_infer_example1() {
        // batch_dims=0, data [2,2], indices [2,2] -> output [2]
        let mut node = create_test_node(2, 2, 2, 0).build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 12, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 1),
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_infer_example2() {
        // batch_dims=0, data [2,2], indices [2,1] -> output [2,2]
        let mut node = create_test_node(2, 2, 1, 0).build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 12, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_infer_example3() {
        // batch_dims=0, data [2,2,2], indices [2,2] -> output [2,2]
        let mut node = create_test_node(3, 2, 2, 0).build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 12, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_infer_example4() {
        // batch_dims=0, data [2,2,2], indices [2,1,2] -> output [2,1,2]
        let mut node = create_test_node(3, 3, 2, 0).build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 12, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 3),
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_infer_example5_batch() {
        // batch_dims=1, data [2,2,2], indices [2,1] -> output [2,2]
        let mut node = create_test_node(3, 2, 1, 1).build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 12, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_infer_scalar_output() {
        // batch_dims=0, data [3], indices [1] (q=1, r=1, k=1) -> rank 0 -> scalar
        let mut node = create_test_node(1, 1, 1, 0).build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 12, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(dtype) => assert_eq!(*dtype, DType::F32),
            other => panic!("Expected scalar, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_batch_dims() {
        // batch_dims=2, data rank=2, indices rank=2 -> b >= min(q,r) should fail
        let mut node = create_test_node(2, 2, 1, 2).build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 12, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_k_too_large() {
        // batch_dims=0, data rank=2, indices last dim=3 -> k > r - b should fail
        let mut node = create_test_node(2, 2, 3, 0).build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 12, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_unexpected_attribute() {
        let node = create_test_node(2, 2, 2, 0)
            .attr_int("unknown_attr", 42)
            .build();
        let processor = GatherNDProcessor;
        let prefs = OutputPreferences::new();
        let mut node = node;
        let result = processor.infer_types(&mut node, 12, &prefs);
        assert!(result.is_err());
    }
}
