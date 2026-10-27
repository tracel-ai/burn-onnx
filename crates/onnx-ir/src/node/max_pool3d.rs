//! # MaxPool (3D)
//!
//! 3D max pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MaxPool.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial MaxPool operator
//! - **Opset 8**: Added storage_order attribute
//! - **Opset 10**: Added ceil_mode and dilations attributes
//! - **Opset 11**: Updated padding semantics
//! - **Opset 12**: Added int8/uint8 dtype support
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, DType, Node, RawNode, TensorType};
use crate::node::padding::{AutoPad, PaddingConfig3d, padding_config_3d};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for MaxPool3d operations
#[derive(Debug, Clone, new)]
pub struct MaxPool3dConfig {
    /// Kernel size [depth, height, width]
    pub kernel_size: [usize; 3],
    /// Stride [depth, height, width]
    pub strides: [usize; 3],
    /// Padding configuration
    pub padding: PaddingConfig3d,
    /// Dilation [depth, height, width]
    pub dilation: [usize; 3],
    /// Whether to use ceil mode for output size calculation (opset 10+)
    pub ceil_mode: bool,
    /// Auto padding mode
    pub auto_pad: AutoPad,
    /// Layout of the optional Indices output: 0 flattens row-major (default), 1
    /// column-major within each spatial volume.
    #[new(default)]
    pub storage_order: i64,
}

/// Node representation for MaxPool3d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MaxPool3dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: MaxPool3dConfig,
}

pub(crate) struct MaxPool3dProcessor;

impl NodeProcessor for MaxPool3dProcessor {
    type Config = MaxPool3dConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Range(1, 2),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" => {}
                "storage_order" => {}
                "dilations" => {
                    let dilations = value.clone().into_i64s()?;
                    if dilations.iter().any(|&d| d != 1) && opset < 10 {
                        return Err(ProcessError::Custom(format!(
                            "MaxPool: dilation requires opset 10+, got opset {}",
                            opset
                        )));
                    }
                }
                "auto_pad" => {
                    AutoPad::parse(&value.clone().into_string()?)?;
                }
                "ceil_mode" => {
                    let ceil_mode = value.clone().into_i64()?;
                    if ceil_mode != 0 && opset < 10 {
                        return Err(ProcessError::Custom(format!(
                            "MaxPool: ceil_mode requires opset 10+, got opset {}",
                            opset
                        )));
                    }
                }
                _ => {}
            }
        }

        let tensor = match &node.inputs[0].ty {
            crate::ir::ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        if tensor.rank != 5 {
            return Err(ProcessError::Custom(format!(
                "MaxPool3d expects input tensor of rank 5 (N x C x D x H x W), got rank {}",
                tensor.rank
            )));
        }

        crate::node::padding::pool_output_type(node)?;

        // The optional Indices output holds int64 positions into the flattened input.
        if let Some(indices) = node.outputs.get_mut(1) {
            indices.ty = ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 5,
                static_shape: None,
            });
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1, 1];
        let mut pads = vec![0, 0, 0, 0, 0, 0];
        let mut dilations = vec![1, 1, 1];
        let mut ceil_mode: i64 = 0;
        let mut auto_pad = AutoPad::NotSet;
        let mut storage_order = 0;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s()?,
                "strides" => strides = value.clone().into_i64s()?,
                "pads" => pads = value.clone().into_i64s()?,
                "dilations" => dilations = value.clone().into_i64s()?,
                "ceil_mode" => ceil_mode = value.clone().into_i64()?,
                "auto_pad" => auto_pad = AutoPad::parse(&value.clone().into_string()?)?,
                "storage_order" => storage_order = value.clone().into_i64()?,
                _ => {}
            }
        }

        if !matches!(storage_order, 0 | 1) {
            return Err(ProcessError::InvalidAttribute {
                name: "storage_order".to_string(),
                reason: format!("expected 0 (row major) or 1 (column major), got {storage_order}"),
            });
        }

        crate::node::padding::check_attr_len("kernel_shape", &kernel_shape, 3)?;
        crate::node::padding::check_attr_len("strides", &strides, 3)?;
        crate::node::padding::check_attr_len("dilations", &dilations, 3)?;

        let padding = padding_config_3d(&pads)?;

        let mut config = MaxPool3dConfig::new(
            [
                kernel_shape[0] as usize,
                kernel_shape[1] as usize,
                kernel_shape[2] as usize,
            ],
            [
                strides[0] as usize,
                strides[1] as usize,
                strides[2] as usize,
            ],
            padding,
            [
                dilations[0] as usize,
                dilations[1] as usize,
                dilations[2] as usize,
            ],
            ceil_mode == 1,
            auto_pad,
        );
        config.storage_order = storage_order;

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::MaxPool3d(MaxPool3dNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        ceil_mode: i64,
        auto_pad: Option<&str>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::MaxPool3d, "test_maxpool3d")
            .input_tensor_f32("data", 5, None)
            .output_tensor_f32("output", 5, None)
            .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_int("ceil_mode", ceil_mode)
            .attr_ints("dilations", dilations);
        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }
        builder.build()
    }

    #[test]
    fn test_max_pool3d_indices_output_is_i64() {
        // Declared as f32 so the test shows infer_types retypes it.
        let mut node = TestNodeBuilder::new(NodeType::MaxPool3d, "test_indices")
            .input_tensor_f32("data", 5, None)
            .output_tensor_f32("output", 5, None)
            .output_tensor_f32("indices", 5, None)
            .attr_ints("kernel_shape", vec![2, 2, 2])
            .build();
        MaxPool3dProcessor
            .infer_types(&mut node, 12, &OutputPreferences::new())
            .unwrap();
        assert!(matches!(
            node.outputs[1].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 5,
                ..
            })
        ));
    }

    #[test]
    fn test_max_pool3d_storage_order() {
        // Default row-major, column-major parsed, anything else rejected
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0; 6],
            vec![1, 1, 1],
            0,
            None,
        );
        let config = MaxPool3dProcessor.extract_config(&node, 12).unwrap();
        assert_eq!(config.storage_order, 0);

        let mut node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0; 6],
            vec![1, 1, 1],
            0,
            None,
        );
        node.attrs.insert(
            "storage_order".to_string(),
            crate::ir::AttributeValue::Int64(1),
        );
        let config = MaxPool3dProcessor.extract_config(&node, 12).unwrap();
        assert_eq!(config.storage_order, 1);

        node.attrs.insert(
            "storage_order".to_string(),
            crate::ir::AttributeValue::Int64(2),
        );
        let err = MaxPool3dProcessor.extract_config(&node, 12).unwrap_err();
        assert!(
            matches!(&err, ProcessError::InvalidAttribute { name, .. } if name == "storage_order"),
            "{err}"
        );
    }

    #[test]
    fn test_max_pool3d_config_basic() {
        let mut node = create_test_node(
            vec![3, 3, 3],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            0,
            None,
        );
        let processor = MaxPool3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3, 3]);
        assert_eq!(config.strides, [1, 1, 1]);
        assert_eq!(config.dilation, [1, 1, 1]);
        assert!(!config.ceil_mode);
        assert!(matches!(config.padding, PaddingConfig3d::Valid));
    }

    #[test]
    fn test_max_pool3d_config_with_padding() {
        let mut node = create_test_node(
            vec![2, 2, 2],
            vec![2, 2, 2],
            vec![1, 1, 1, 1, 1, 1],
            vec![1, 1, 1],
            0,
            None,
        );
        let processor = MaxPool3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(
            config.padding,
            PaddingConfig3d::Explicit(1, 1, 1, 1, 1, 1)
        ));
    }

    #[test]
    fn test_max_pool3d_dilation_opset_validation() {
        // Test that non-default dilations at opset < 10 are rejected
        let mut node = create_test_node(
            vec![3, 3, 3],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![2, 2, 2],
            0,
            None,
        );
        let processor = MaxPool3dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 9, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_max_pool3d_ceil_mode_opset_validation() {
        let mut node = create_test_node(
            vec![3, 3, 3],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            None,
        );
        let processor = MaxPool3dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 9, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_max_pool3d_rejects_non_5d_input() {
        let mut node = TestNodeBuilder::new(NodeType::MaxPool3d, "test_maxpool3d_bad")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", vec![3, 3, 3])
            .attr_ints("strides", vec![1, 1, 1])
            .attr_ints("pads", vec![0, 0, 0, 0, 0, 0])
            .attr_ints("dilations", vec![1, 1, 1])
            .attr_int("ceil_mode", 0)
            .build();
        let processor = MaxPool3dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
