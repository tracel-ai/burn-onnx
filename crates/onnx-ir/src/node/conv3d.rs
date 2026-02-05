//! # Conv (3D)
//!
//! 3D convolution operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Conv.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic convolution support
//! - **Opset 11**: No changes to Conv operator itself (broader ONNX updates)

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};

use crate::node::padding::{AutoPad, PaddingConfig3d, padding_config_3d};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for Conv3d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct Conv3dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: Conv3dConfig,
}

/// Configuration for Conv3d operations.
#[derive(Debug, Clone, PartialEq, Eq, new)]
#[allow(clippy::too_many_arguments)]
pub struct Conv3dConfig {
    /// Size of the kernel.
    pub kernel_size: [usize; 3],
    /// Stride of the convolutional kernel.
    pub stride: [usize; 3],
    /// Dilation of the convolutional kernel.
    pub dilation: [usize; 3],
    /// Groups.
    pub groups: usize,
    /// Padding.
    pub padding: PaddingConfig3d,
    /// Auto padding mode
    pub auto_pad: AutoPad,
}

pub(crate) struct Conv3dProcessor;

impl NodeProcessor for Conv3dProcessor {
    type Config = Conv3dConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Range(2, 3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift weight (input[1]) and optional bias (input[2])
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Output type is same as input
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1, 1];
        let mut pads = vec![0, 0, 0, 0, 0, 0];
        let mut dilations = vec![1, 1, 1];
        let mut group: usize = 1;
        let mut auto_pad = AutoPad::NotSet;

        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("Conv3d: weight tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "group" => group = value.clone().into_i64() as usize,
                "auto_pad" => {
                    auto_pad = AutoPad::parse(&value.clone().into_string())?;
                }
                _ => {
                    // TODO: According to spec, there may be other valid attributes that are not handled
                    // Consider logging/warning instead of rejecting unknown attributes
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Conv3d: {key}"),
                    });
                }
            }
        }

        let padding = padding_config_3d(&pads);

        let kernel_size = if kernel_shape.is_empty() {
            // Spec says if kernel shape not present in attributes it should be inferred from the weight tensor
            if weight_shape.len() != 5 {
                return Err(ProcessError::Custom(format!(
                    "expected to infer kernel shape from a weight tensor of rank 5 but got shape {weight_shape:?}"
                )));
            }

            [weight_shape[2], weight_shape[3], weight_shape[4]]
        } else {
            [
                kernel_shape[0] as _,
                kernel_shape[1] as _,
                kernel_shape[2] as _,
            ]
        };

        let config = Conv3dConfig::new(
            kernel_size,
            [
                strides[0] as usize,
                strides[1] as usize,
                strides[2] as usize,
            ],
            [
                dilations[0] as usize,
                dilations[1] as usize,
                dilations[2] as usize,
            ],
            group,
            padding,
            auto_pad,
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Conv3d(Conv3dNode {
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

    fn create_test_node(
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
        has_bias: bool,
        auto_pad: Option<&str>,
    ) -> TestNodeBuilder {
        // Create weight tensor data (not important for the test)
        let weight_shape = vec![4, 2, 2, 2, 2]; // [output_channels, input_channels/groups, k_d, k_h, k_w]
        let weight_data = vec![0.0; 64]; // 4*2*2*2*2 = 64

        let has_kernel_shape = !kernel_shape.is_empty();

        // Start building the node with input and weight
        let mut builder = TestNodeBuilder::new(NodeType::Conv3d, "test_conv3d")
            .input_tensor_f32("data", 5, None)
            .input_tensor_f32_data("weight", weight_data, weight_shape)
            .output_tensor_f32("output", 5, None);

        // Add bias if needed
        if has_bias {
            builder = builder.input_tensor_f32("bias", 1, None);
        }

        // Add attributes
        builder = builder
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_int("group", group);

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }

        builder
    }

    #[test]
    fn test_conv3d_config_basic() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // channels removed from config (derived in burn-onnx)
        assert_eq!(config.kernel_size, [2, 2, 2]);
        assert_eq!(config.stride, [1, 1, 1]);
        assert_eq!(config.dilation, [1, 1, 1]);
        assert_eq!(config.groups, 1);
        // bias removed from config (derived in burn-onnx)
        assert!(matches!(config.padding, PaddingConfig3d::Valid));
    }

    #[test]
    fn test_conv3d_config_with_padding() {
        let node = create_test_node(
            vec![3, 3, 3],
            vec![1, 1, 1],
            vec![1, 1, 1, 1, 1, 1],
            vec![1, 1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3, 3]);
        assert!(matches!(
            config.padding,
            PaddingConfig3d::Explicit(1, 1, 1, 1, 1, 1)
        ));
    }

    #[test]
    fn test_conv3d_config_with_groups() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            2,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.groups, 2);
        // channels removed from config (derived in burn-onnx)
    }

    #[test]
    fn test_conv3d_config_autopad_not_set() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // channels removed from config (derived in burn-onnx)
        assert_eq!(config.kernel_size, [2, 2, 2]);
        assert_eq!(config.stride, [1, 1, 1]);
        assert_eq!(config.dilation, [1, 1, 1]);
        assert_eq!(config.groups, 1);
        // bias removed from config (derived in burn-onnx)
        assert!(matches!(config.padding, PaddingConfig3d::Valid));
    }

    #[test]
    fn test_conv3d_config_autopad_same_upper() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(16);
        let processor = Conv3dProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.auto_pad, AutoPad::SameUpper);
    }

    #[test]
    fn test_conv3d_config_kernel_shape_not_set() {
        let node = create_test_node(
            vec![],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [2, 2, 2]); // Inferred via weight tensor shape
    }
}
