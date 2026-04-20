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
                    }
                }
                "offset" => {
                    if let AttributeValue::Float32s(floats) = value {
                        offset = Some(floats.clone());
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
}
