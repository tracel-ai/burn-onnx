use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::proto_conversion::element_type_from_proto;

#[derive(Debug, Clone, Default)]
pub struct QuantizeLinearConfig {
    pub axis: Option<i64>,
    pub block_size: Option<i64>,
    pub output_dtype: Option<DType>,
    pub precision: Option<DType>,
    pub saturate: Option<i64>,
}

#[derive(Debug, Clone, NodeBuilder)]
pub struct QuantizeLinearNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: QuantizeLinearConfig,
}

pub(crate) struct QuantizeLinearProcessor;

impl NodeProcessor for QuantizeLinearProcessor {
    type Config = QuantizeLinearConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Range(2, 3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let config = self.extract_config(node, opset)?;

        if config.saturate.is_some() && opset < 19 {
            return Err(ProcessError::Custom(format!(
                "QuantizeLinear: saturate requires opset 19+, got {opset}"
            )));
        }

        if let Some(block_size) = config.block_size
            && block_size > 0
        {
            return Err(ProcessError::Custom(format!(
                "QuantizeLinear: blocked quantization (block_size={block_size}) is not supported yet"
            )));
        }

        if !node.inputs[0].ty.is_on_device() {
            return Err(ProcessError::TypeMismatch {
                expected: "on-device tensor for x".to_string(),
                actual: format!("{:?}", node.inputs[0].ty),
            });
        }

        if !node.inputs[1].ty.is_on_device() {
            return Err(ProcessError::TypeMismatch {
                expected: "on-device tensor for y_scale".to_string(),
                actual: format!("{:?}", node.inputs[1].ty),
            });
        }

        let x_dtype = node.inputs[0].ty.elem_type();
        if !(x_dtype.is_float() || x_dtype.is_int()) {
            return Err(ProcessError::TypeMismatch {
                expected: "float or int tensor for x".to_string(),
                actual: format!("{:?}", x_dtype),
            });
        }

        let scale_dtype = node.inputs[1].ty.elem_type();
        if !(scale_dtype.is_float() || scale_dtype == DType::I32) {
            return Err(ProcessError::TypeMismatch {
                expected: "float or int32 tensor for y_scale".to_string(),
                actual: format!("{:?}", scale_dtype),
            });
        }

        let zero_dtype = node.get_input(2).map(|arg| arg.ty.elem_type());
        let output_dtype = if let Some(dtype) = config.output_dtype {
            if let Some(zero) = zero_dtype
                && zero != dtype
            {
                return Err(ProcessError::TypeMismatch {
                    expected: format!("y_zero_point dtype {:?}", dtype),
                    actual: format!("{:?}", zero),
                });
            }
            dtype
        } else if let Some(zero) = zero_dtype {
            zero
        } else {
            DType::U8
        };

        let supported = matches!(
            output_dtype,
            DType::U8 | DType::I8 | DType::U16 | DType::I16
        );
        if !supported {
            return Err(ProcessError::Custom(format!(
                "QuantizeLinear: output dtype {:?} is not supported yet",
                output_dtype
            )));
        }

        node.outputs[0].ty = match node.inputs[0].ty.clone() {
            ArgType::Tensor(tensor) => ArgType::Tensor(TensorType {
                dtype: output_dtype,
                rank: tensor.rank,
                static_shape: tensor.static_shape,
            }),
            ArgType::ScalarTensor(_) => ArgType::ScalarTensor(output_dtype),
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "tensor/scalar tensor input".to_string(),
                    actual: format!("{:?}", other),
                });
            }
        };

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut config = QuantizeLinearConfig::default();

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => {
                    config.axis = Some(value.clone().into_i64());
                }
                "block_size" => {
                    config.block_size = Some(value.clone().into_i64());
                }
                "output_dtype" => {
                    let dtype = match value {
                        AttributeValue::Int64(type_id) => element_type_from_proto(*type_id as i32)
                            .map_err(|_| ProcessError::InvalidAttribute {
                                name: "output_dtype".to_string(),
                                reason: format!("unsupported dtype: {type_id}"),
                            })?,
                        _ => {
                            return Err(ProcessError::InvalidAttribute {
                                name: "output_dtype".to_string(),
                                reason: "must be Int64".to_string(),
                            });
                        }
                    };
                    config.output_dtype = Some(dtype);
                }
                "precision" => {
                    let dtype = match value {
                        AttributeValue::Int64(type_id) => element_type_from_proto(*type_id as i32)
                            .map_err(|_| ProcessError::InvalidAttribute {
                                name: "precision".to_string(),
                                reason: format!("unsupported dtype: {type_id}"),
                            })?,
                        _ => {
                            return Err(ProcessError::InvalidAttribute {
                                name: "precision".to_string(),
                                reason: "must be Int64".to_string(),
                            });
                        }
                    };
                    config.precision = Some(dtype);
                }
                "saturate" => {
                    config.saturate = Some(value.clone().into_i64());
                }
                _ => {}
            }
        }

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::QuantizeLinear(QuantizeLinearNode {
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
    use crate::protos::tensor_proto::DataType;
    use protobuf::Enum;

    #[test]
    fn test_quantize_linear_default_output_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::QuantizeLinear, "q")
            .input_tensor_f32("x", 2, None)
            .input_tensor_f32("y_scale", 0, None)
            .output_tensor_f32("y", 2, None)
            .build();

        let processor = QuantizeLinearProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        assert_eq!(node.outputs[0].ty.elem_type(), DType::U8);
    }

    #[test]
    fn test_quantize_linear_uses_zero_point_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::QuantizeLinear, "q")
            .input_tensor_f32("x", 2, None)
            .input_tensor_f32("y_scale", 0, None)
            .input_tensor_i32("y_zero_point", 0, None)
            .output_tensor_f32("y", 2, None)
            .build();

        node.inputs[2].ty = ArgType::Tensor(TensorType::new(DType::I8, 0, None));

        let processor = QuantizeLinearProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        assert_eq!(node.outputs[0].ty.elem_type(), DType::I8);
    }

    #[test]
    fn test_quantize_linear_output_dtype_must_match_zero_point() {
        let mut node = TestNodeBuilder::new(NodeType::QuantizeLinear, "q")
            .input_tensor_f32("x", 2, None)
            .input_tensor_f32("y_scale", 0, None)
            .input_tensor_i32("y_zero_point", 0, None)
            .output_tensor_f32("y", 2, None)
            .attr_int("output_dtype", DataType::UINT8.value() as i64)
            .build();

        node.inputs[2].ty = ArgType::Tensor(TensorType::new(DType::I8, 0, None));

        let processor = QuantizeLinearProcessor;
        let err = processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_quantize_linear_rejects_blocked_quantization() {
        let mut node = TestNodeBuilder::new(NodeType::QuantizeLinear, "q")
            .input_tensor_f32("x", 2, None)
            .input_tensor_f32("y_scale", 0, None)
            .output_tensor_f32("y", 2, None)
            .attr_int("block_size", 8)
            .build();

        let processor = QuantizeLinearProcessor;
        let err = processor
            .infer_types(&mut node, 21, &OutputPreferences::new())
            .unwrap_err();

        let err_msg = format!("{}", err);
        assert!(err_msg.contains("blocked quantization"));
    }
}
