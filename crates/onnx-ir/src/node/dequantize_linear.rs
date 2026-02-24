use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::proto_conversion::element_type_from_proto;

#[derive(Debug, Clone, Default)]
pub struct DequantizeLinearConfig {
    pub axis: Option<i64>,
    pub block_size: Option<i64>,
    pub output_dtype: Option<DType>,
}

#[derive(Debug, Clone, NodeBuilder)]
pub struct DequantizeLinearNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: DequantizeLinearConfig,
}

pub(crate) struct DequantizeLinearProcessor;

impl NodeProcessor for DequantizeLinearProcessor {
    type Config = DequantizeLinearConfig;

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

        if config.output_dtype.is_some() && opset < 24 {
            return Err(ProcessError::Custom(format!(
                "DequantizeLinear: output_dtype requires opset 24+, got {opset}"
            )));
        }

        if let Some(block_size) = config.block_size
            && block_size > 0
        {
            return Err(ProcessError::Custom(format!(
                "DequantizeLinear: blocked quantization (block_size={block_size}) is not supported yet"
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
                expected: "on-device tensor for x_scale".to_string(),
                actual: format!("{:?}", node.inputs[1].ty),
            });
        }

        let x_dtype = node.inputs[0].ty.elem_type();
        if !(x_dtype.is_int() || x_dtype.is_uint()) {
            return Err(ProcessError::TypeMismatch {
                expected: "integer tensor for x".to_string(),
                actual: format!("{:?}", x_dtype),
            });
        }

        let scale_dtype = node.inputs[1].ty.elem_type();
        if !scale_dtype.is_float() {
            return Err(ProcessError::TypeMismatch {
                expected: "floating point tensor for x_scale".to_string(),
                actual: format!("{:?}", scale_dtype),
            });
        }

        if let Some(zero) = node.get_input(2) {
            if !zero.ty.is_on_device() {
                return Err(ProcessError::TypeMismatch {
                    expected: "on-device tensor for x_zero_point".to_string(),
                    actual: format!("{:?}", zero.ty),
                });
            }

            let zero_dtype = zero.ty.elem_type();
            if zero_dtype != x_dtype {
                return Err(ProcessError::TypeMismatch {
                    expected: format!("x_zero_point dtype {:?}", x_dtype),
                    actual: format!("{:?}", zero_dtype),
                });
            }
        }

        let output_dtype = config.output_dtype.unwrap_or(scale_dtype);
        if !output_dtype.is_float() {
            return Err(ProcessError::TypeMismatch {
                expected: "floating point output dtype".to_string(),
                actual: format!("{:?}", output_dtype),
            });
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
        let mut config = DequantizeLinearConfig::default();

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
                _ => {}
            }
        }

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::DequantizeLinear(DequantizeLinearNode {
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
    fn test_dequantize_linear_infer_types_default_output_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::DequantizeLinear, "dq")
            .input_tensor_i32("x", 2, None)
            .input_tensor_f32("x_scale", 0, None)
            .output_tensor_f32("y", 2, None)
            .build();

        node.inputs[0].ty = ArgType::Tensor(TensorType::new(DType::U8, 2, None));

        let processor = DequantizeLinearProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("expected tensor output"),
        }
    }

    #[test]
    fn test_dequantize_linear_output_dtype_attr() {
        let mut node = TestNodeBuilder::new(NodeType::DequantizeLinear, "dq")
            .input_tensor_i32("x", 2, None)
            .input_tensor_f32("x_scale", 0, None)
            .output_tensor_f32("y", 2, None)
            .attr_int("output_dtype", DataType::DOUBLE.value() as i64)
            .build();

        node.inputs[0].ty = ArgType::Tensor(TensorType::new(DType::U8, 2, None));

        let processor = DequantizeLinearProcessor;
        processor
            .infer_types(&mut node, 24, &OutputPreferences::new())
            .unwrap();

        assert_eq!(node.outputs[0].ty.elem_type(), DType::F64);
    }

    #[test]
    fn test_dequantize_linear_zero_point_dtype_validation() {
        let mut node = TestNodeBuilder::new(NodeType::DequantizeLinear, "dq")
            .input_tensor_i32("x", 2, None)
            .input_tensor_f32("x_scale", 0, None)
            .input_tensor_i32("x_zero_point", 0, None)
            .output_tensor_f32("y", 2, None)
            .build();

        node.inputs[0].ty = ArgType::Tensor(TensorType::new(DType::U8, 2, None));
        node.inputs[2].ty = ArgType::Tensor(TensorType::new(DType::I8, 0, None));

        let processor = DequantizeLinearProcessor;
        let err = processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_dequantize_linear_rejects_blocked_quantization() {
        let mut node = TestNodeBuilder::new(NodeType::DequantizeLinear, "dq")
            .input_tensor_i32("x", 2, None)
            .input_tensor_f32("x_scale", 0, None)
            .output_tensor_f32("y", 2, None)
            .attr_int("block_size", 4)
            .build();

        node.inputs[0].ty = ArgType::Tensor(TensorType::new(DType::U8, 2, None));

        let processor = DequantizeLinearProcessor;
        let err = processor
            .infer_types(&mut node, 21, &OutputPreferences::new())
            .unwrap_err();

        let err_msg = format!("{}", err);
        assert!(err_msg.contains("blocked quantization"));
    }
}
