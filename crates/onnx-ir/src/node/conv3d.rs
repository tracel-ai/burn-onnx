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

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};

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
        // Extract input tensor type
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Validate input tensor rank - Conv3d expects rank 5 (N x C x D x H x W)
        if tensor.rank != 5 {
            return Err(ProcessError::Custom(format!(
                "Conv3d expects input tensor of rank 5 (N x C x D x H x W), got rank {}",
                tensor.rank
            )));
        }

        // Validate weight tensor
        let weight_tensor = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor (weight)".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        // Compute output static_shape: [batch, out_channels, D_out, H_out, W_out]
        let static_shape = {
            let batch = tensor
                .static_shape
                .as_ref()
                .and_then(|s| s.first().copied().flatten());
            let out_channels = node.inputs[1]
                .value()
                .and_then(|data| data.shape.first().copied())
                .or_else(|| {
                    weight_tensor
                        .static_shape
                        .as_ref()
                        .and_then(|s| s.first().copied().flatten())
                });

            let compute_spatial = |dim_idx: usize,
                                   kernel: usize,
                                   stride: usize,
                                   dilation: usize,
                                   pad_begin: usize,
                                   pad_end: usize|
             -> Option<usize> {
                let input_dim = tensor
                    .static_shape
                    .as_ref()
                    .and_then(|s| s.get(dim_idx).copied().flatten())?;
                let padding = pad_begin + pad_end;
                let numerator = input_dim as isize + padding as isize
                    - dilation as isize * (kernel as isize - 1)
                    - 1;
                if numerator < 0 {
                    return None;
                }
                Some(numerator as usize / stride + 1)
            };

            let spatial = self.extract_config(node, _opset).ok().map(|config| {
                let (pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right) =
                    config.padding.as_tuple();
                let d_out = compute_spatial(
                    2,
                    config.kernel_size[0],
                    config.stride[0],
                    config.dilation[0],
                    pad_front,
                    pad_back,
                );
                let h_out = compute_spatial(
                    3,
                    config.kernel_size[1],
                    config.stride[1],
                    config.dilation[1],
                    pad_top,
                    pad_bottom,
                );
                let w_out = compute_spatial(
                    4,
                    config.kernel_size[2],
                    config.stride[2],
                    config.dilation[2],
                    pad_left,
                    pad_right,
                );
                (d_out, h_out, w_out)
            });
            let (d_out, h_out, w_out) = spatial.unwrap_or((None, None, None));
            Some(vec![batch, out_channels, d_out, h_out, w_out])
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape,
        });

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

    #[test]
    fn test_conv3d_static_shape_known() {
        // Input [1, 2, 8, 8, 8], weight [4, 2, 2, 2, 2], stride=[1,1,1], pad=0, dilation=[1,1,1]
        // D/H/W_out = (8 + 0 - 1*(2-1) - 1) / 1 + 1 = 7
        let mut node = TestNodeBuilder::new(NodeType::Conv3d, "test")
            .input_tensor_f32("data", 5, Some(vec![1, 2, 8, 8, 8]))
            .input_tensor_f32_data("weight", vec![0.0; 64], vec![4, 2, 2, 2, 2])
            .output_tensor_f32("output", 5, None)
            .attr_ints("kernel_shape", vec![2, 2, 2])
            .attr_ints("strides", vec![1, 1, 1])
            .attr_ints("pads", vec![0, 0, 0, 0, 0, 0])
            .attr_ints("dilations", vec![1, 1, 1])
            .attr_int("group", 1)
            .build_with_graph_data(16);

        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(t) => {
                assert_eq!(t.rank, 5);
                assert_eq!(
                    t.static_shape,
                    Some(vec![Some(1), Some(4), Some(7), Some(7), Some(7)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_conv3d_static_shape_no_input_shape() {
        // No input static_shape -> batch and spatial are None, out_channels known
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(t) => {
                assert_eq!(t.rank, 5);
                assert_eq!(t.static_shape, Some(vec![None, Some(4), None, None, None]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
