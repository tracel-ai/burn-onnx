//! # Scaler
//!
//! Rescales input data by applying the formula: Y = (X - offset) * scale.
//! The Scaler operator is part of the ONNX ML operators for preprocessing.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx_aionnxml_Scaler.html>
//!
//! ## Type Constraints
//!
//! - T: tensor(float), tensor(double), tensor(int32), tensor(int64)
//!
//! ## Opset Versions
//!
//! - **Opset 1**: Initial version with scale and offset attributes

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for Scaler operation
#[derive(Debug, Clone, Default, new)]
pub struct ScalerConfig {
    /// Scaling factor(s) to multiply the input after subtracting the offset
    pub scale: Option<Vec<f32>>,
    /// Offset value(s) to subtract from the input before multiplying by scale
    pub offset: Option<Vec<f32>>,
}

/// Node representation for Scaler operation
#[derive(Debug, Clone, new, NodeBuilder)]
pub struct ScalerNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ScalerConfig,
}

pub(crate) struct ScalerProcessor;

impl NodeProcessor for ScalerProcessor {
    type Config = ScalerConfig;

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
        // Per ONNX spec the output Y is always tensor(float) regardless of input dtype.
        // Input T must be tensor(float), tensor(double), tensor(int32), or tensor(int64).
        // Scaler is shape-preserving, so copy rank and static_shape from the input.
        let (rank, static_shape) = match &node.inputs[0].ty {
            ArgType::Tensor(t) => {
                match t.dtype {
                    DType::F32 | DType::F64 | DType::I32 | DType::I64 => {}
                    other => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "tensor(float | double | int32 | int64)".to_string(),
                            actual: format!("tensor({other:?})"),
                        });
                    }
                }
                (t.rank, t.static_shape.clone())
            }
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "tensor(float | double | int32 | int64)".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::F32,
            rank,
            static_shape,
        });
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut scale: Option<Vec<f32>> = None;
        let mut offset: Option<Vec<f32>> = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "scale" => {
                    if let AttributeValue::Float32s(floats) = value {
                        scale = Some(floats.clone());
                    } else {
                        return Err(ProcessError::InvalidAttribute {
                            name: "scale".to_string(),
                            reason: format!("expected Float32s, got {value:?}"),
                        });
                    }
                }
                "offset" => {
                    if let AttributeValue::Float32s(floats) = value {
                        offset = Some(floats.clone());
                    } else {
                        return Err(ProcessError::InvalidAttribute {
                            name: "offset".to_string(),
                            reason: format!("expected Float32s, got {value:?}"),
                        });
                    }
                }
                _ => {}
            }
        }

        if let (Some(s), Some(o)) = (&scale, &offset)
            && s.len() != o.len()
        {
            return Err(ProcessError::InvalidAttribute {
                name: "scale/offset".to_string(),
                reason: format!(
                    "scale and offset must have the same length, got {} and {}",
                    s.len(),
                    o.len()
                ),
            });
        }

        Ok(ScalerConfig::new(scale, offset))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        // extract_config is infallible here: any invalid attribute would have
        // already caused extract_config to fail during the type-inference pass.
        let config = self
            .extract_config(&builder, opset)
            .expect("ScalerProcessor: config extraction failed");
        Node::Scaler(ScalerNode::new(
            builder.name,
            builder.inputs,
            builder.outputs,
            config,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;
    use burn_tensor::BoolStore;

    fn make_node(scale: Option<Vec<f32>>, offset: Option<Vec<f32>>, dtype: DType) -> RawNode {
        let mut builder =
            match dtype {
                DType::F32 => TestNodeBuilder::new(NodeType::Scaler, "test_scaler")
                    .input_tensor_f32("X", 2, None),
                DType::F64 => TestNodeBuilder::new(NodeType::Scaler, "test_scaler")
                    .input_tensor_f64("X", 2, None),
                DType::I32 => TestNodeBuilder::new(NodeType::Scaler, "test_scaler")
                    .input_tensor_i32("X", 2, None),
                DType::I64 => TestNodeBuilder::new(NodeType::Scaler, "test_scaler")
                    .input_tensor_i64("X", 2, None),
                DType::Bool(_) => TestNodeBuilder::new(NodeType::Scaler, "test_scaler")
                    .input_tensor_bool("X", 2, None),
                _ => panic!("unsupported dtype in test helper"),
            }
            .output_tensor_f32("Y", 2, None);
        if let Some(s) = scale {
            builder = builder.attr_floats("scale", s);
        }
        if let Some(o) = offset {
            builder = builder.attr_floats("offset", o);
        }
        builder.build()
    }

    #[test]
    fn test_scaler_config_extraction() {
        let config = ScalerConfig::new(Some(vec![2.0]), Some(vec![1.0]));
        assert!(config.scale.is_some());
        assert_eq!(config.scale.unwrap(), vec![2.0]);
        assert!(config.offset.is_some());
        assert_eq!(config.offset.unwrap(), vec![1.0]);
    }

    #[test]
    fn test_scaler_node_builder() {
        let config = ScalerConfig::new(Some(vec![2.0]), Some(vec![1.0]));
        let node = ScalerNode::new("test_scaler".to_string(), vec![], vec![], config);

        assert_eq!(node.name, "test_scaler");
        assert_eq!(node.inputs.len(), 0);
        assert_eq!(node.outputs.len(), 0);
        assert!(node.config.scale.is_some());
        assert!(node.config.offset.is_some());
    }

    #[test]
    fn test_infer_types_f32_preserves_shape() {
        let mut node = make_node(Some(vec![2.0]), Some(vec![1.0]), DType::F32);
        let processor = ScalerProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 1, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.rank, 2);
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn test_infer_types_int64_output_is_f32() {
        let mut node = make_node(Some(vec![2.0]), None, DType::I64);
        let processor = ScalerProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 1, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.dtype, DType::F32),
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn test_infer_types_rejects_invalid_dtype() {
        let mut node = make_node(None, None, DType::Bool(BoolStore::Native));
        let processor = ScalerProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 1, &prefs).unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_extract_config_both_attrs() {
        let node = make_node(Some(vec![2.0, 3.0]), Some(vec![0.5, 1.0]), DType::F32);
        let processor = ScalerProcessor;
        let config = processor.extract_config(&node, 1).unwrap();
        assert_eq!(config.scale.unwrap(), vec![2.0, 3.0]);
        assert_eq!(config.offset.unwrap(), vec![0.5, 1.0]);
    }

    #[test]
    fn test_extract_config_length_mismatch_error() {
        let node = make_node(Some(vec![1.0, 2.0]), Some(vec![0.5]), DType::F32);
        let processor = ScalerProcessor;
        let err = processor.extract_config(&node, 1).unwrap_err();
        assert!(matches!(
            err,
            ProcessError::InvalidAttribute { ref name, .. } if name == "scale/offset"
        ));
    }
}
