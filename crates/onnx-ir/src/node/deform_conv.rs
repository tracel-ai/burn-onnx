//! # DeformConv
//!
//! Deformable convolution operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__DeformConv.html>
//!
//! ## Opset Versions
//! - **Opset 19**: Initial version

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::node::padding::{PaddingConfig2d, padding_config_2d};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for DeformConv operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct DeformConvNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: DeformConvConfig,
}

/// Configuration for DeformConv operations
#[derive(Debug, Clone, new)]
#[allow(clippy::too_many_arguments)]
pub struct DeformConvConfig {
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub stride: [usize; 2],
    /// Padding configuration
    pub padding: PaddingConfig2d,
    /// Dilation [height, width]
    pub dilation: [usize; 2],
    /// Number of weight groups
    pub groups: usize,
    /// Number of offset groups
    pub offset_groups: usize,
}

/// Node processor for DeformConv operation
pub(crate) struct DeformConvProcessor;

impl NodeProcessor for DeformConvProcessor {
    type Config = DeformConvConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 19,
            max_opset: None,
            inputs: InputSpec::Range(3, 5),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift weight (input[1]) to static
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        // Lift optional bias (input[3]) to static
        if node.inputs.len() > 3 && !node.inputs[3].is_optional() && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate input X (rank 4)
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };
        if tensor.rank != 4 {
            return Err(ProcessError::Custom(format!(
                "DeformConv expects input tensor of rank 4 (N x C x H x W), got rank {}",
                tensor.rank
            )));
        }

        // Validate weight W (rank 4)
        let weight_tensor = match &node.inputs[1].ty {
            ArgType::Tensor(t) => t,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor (weight)".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };
        if weight_tensor.rank != 4 {
            return Err(ProcessError::Custom(format!(
                "DeformConv expects weight tensor of rank 4 (oC x C/group x kH x kW), got rank {}",
                weight_tensor.rank
            )));
        }

        // Validate offset (rank 4)
        let offset_tensor = match &node.inputs[2].ty {
            ArgType::Tensor(t) => t,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor (offset)".to_string(),
                    actual: format!("{:?}", node.inputs[2].ty),
                });
            }
        };
        if offset_tensor.rank != 4 {
            return Err(ProcessError::Custom(format!(
                "DeformConv expects offset tensor of rank 4, got rank {}",
                offset_tensor.rank
            )));
        }

        // Validate optional bias (rank 1)
        if let Some(bias_arg) = node.get_input(3) {
            let bias_tensor = match &bias_arg.ty {
                ArgType::Tensor(t) => t,
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor (bias)".to_string(),
                        actual: format!("{:?}", bias_arg.ty),
                    });
                }
            };
            if bias_tensor.rank != 1 {
                return Err(ProcessError::Custom(format!(
                    "DeformConv expects bias tensor of rank 1, got rank {}",
                    bias_tensor.rank
                )));
            }
        }

        // Validate optional mask (rank 4)
        if let Some(mask_arg) = node.get_input(4) {
            let mask_tensor = match &mask_arg.ty {
                ArgType::Tensor(t) => t,
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor (mask)".to_string(),
                        actual: format!("{:?}", mask_arg.ty),
                    });
                }
            };
            if mask_tensor.rank != 4 {
                return Err(ProcessError::Custom(format!(
                    "DeformConv expects mask tensor of rank 4, got rank {}",
                    mask_tensor.rank
                )));
            }
        }

        // Validate dtype consistency: W, offset, bias, mask must match X
        let expected_dtype = tensor.dtype;
        if weight_tensor.dtype != expected_dtype {
            return Err(ProcessError::Custom(format!(
                "DeformConv: weight dtype {:?} does not match input dtype {:?}",
                weight_tensor.dtype, expected_dtype
            )));
        }
        if offset_tensor.dtype != expected_dtype {
            return Err(ProcessError::Custom(format!(
                "DeformConv: offset dtype {:?} does not match input dtype {:?}",
                offset_tensor.dtype, expected_dtype
            )));
        }
        if let Some(bias_arg) = node.get_input(3)
            && let ArgType::Tensor(bt) = &bias_arg.ty
            && bt.dtype != expected_dtype
        {
            return Err(ProcessError::Custom(format!(
                "DeformConv: bias dtype {:?} does not match input dtype {:?}",
                bt.dtype, expected_dtype
            )));
        }
        if let Some(mask_arg) = node.get_input(4)
            && let ArgType::Tensor(mt) = &mask_arg.ty
            && mt.dtype != expected_dtype
        {
            return Err(ProcessError::Custom(format!(
                "DeformConv: mask dtype {:?} does not match input dtype {:?}",
                mt.dtype, expected_dtype
            )));
        }

        // Compute output static_shape: [batch, out_channels, H_out, W_out]
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
                if numerator < 0 || stride == 0 {
                    return None;
                }
                Some(numerator as usize / stride + 1)
            };

            let spatial = self.extract_config(node, _opset).ok().map(|config| {
                let (pad_top, pad_left, pad_bottom, pad_right) = config.padding.as_tuple();
                let h_out = compute_spatial(
                    2,
                    config.kernel_size[0],
                    config.stride[0],
                    config.dilation[0],
                    pad_top,
                    pad_bottom,
                );
                let w_out = compute_spatial(
                    3,
                    config.kernel_size[1],
                    config.stride[1],
                    config.dilation[1],
                    pad_left,
                    pad_right,
                );
                (h_out, w_out)
            });
            let (h_out, w_out) = spatial.unwrap_or((None, None));
            Some(vec![batch, out_channels, h_out, w_out])
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: 4,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1];
        let mut pads = vec![0, 0, 0, 0];
        let mut dilations = vec![1, 1];
        let mut group: usize = 1;
        let mut offset_group: usize = 1;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "group" => group = value.clone().into_i64() as usize,
                "offset_group" => offset_group = value.clone().into_i64() as usize,
                _ => {}
            }
        }

        let padding = padding_config_2d(&pads);

        // Only require weight shape when kernel_shape attribute is absent
        let kernel_size = if kernel_shape.is_empty() {
            let weight_shape = node.inputs[1]
                .value()
                .map(|v| v.shape.to_vec())
                .or_else(|| {
                    if let ArgType::Tensor(t) = &node.inputs[1].ty {
                        t.static_shape_known().map(|s| s.to_vec())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    ProcessError::Custom(
                        "DeformConv: kernel_shape attribute missing and weight shape is unknown"
                            .to_string(),
                    )
                })?;
            if weight_shape.len() != 4 {
                return Err(ProcessError::Custom(format!(
                    "DeformConv: expected weight tensor of rank 4 but got shape {weight_shape:?}",
                )));
            }
            [weight_shape[2], weight_shape[3]]
        } else {
            [kernel_shape[0] as _, kernel_shape[1] as _]
        };

        Ok(DeformConvConfig::new(
            kernel_size,
            [strides[0] as usize, strides[1] as usize],
            padding,
            [dilations[0] as usize, dilations[1] as usize],
            group,
            offset_group,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self.extract_config(&builder, opset).unwrap_or_else(|e| {
            panic!(
                "DeformConv '{}' config extraction failed: {e}",
                builder.name
            )
        });

        Node::DeformConv(DeformConvNode {
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
        offset_group: i64,
        has_bias: bool,
        has_mask: bool,
    ) -> TestNodeBuilder {
        // Weight tensor: [output_channels, input_channels/groups, k_h, k_w]
        let weight_shape = vec![4, 2, 2, 2];
        let weight_data = vec![0.0; 32]; // 4*2*2*2 = 32

        let has_kernel_shape = !kernel_shape.is_empty();

        let mut builder = TestNodeBuilder::new(NodeType::DeformConv, "test_deform_conv")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data("weight", weight_data, weight_shape)
            .input_tensor_f32("offset", 4, None)
            .output_tensor_f32("output", 4, None)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_int("group", group)
            .attr_int("offset_group", offset_group);

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        if has_bias {
            builder = builder.input_tensor_f32("bias", 1, None);
        } else {
            builder = builder.add_input("", ArgType::default());
        }

        if has_mask {
            builder = builder.input_tensor_f32("mask", 4, None);
        }

        builder
    }

    #[test]
    fn test_deform_conv_config_basic() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            1,
            false,
            false,
        )
        .build_with_graph_data(19);
        let mut node = node;
        let processor = DeformConvProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 19).unwrap();
        processor.infer_types(&mut node, 19, &prefs).unwrap();

        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.stride, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert_eq!(config.groups, 1);
        assert_eq!(config.offset_groups, 1);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_deform_conv_config_with_padding() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1],
            1,
            1,
            false,
            false,
        )
        .build_with_graph_data(19);
        let processor = DeformConvProcessor;
        let config = processor.extract_config(&node, 19).unwrap();

        assert!(matches!(
            config.padding,
            PaddingConfig2d::Explicit(1, 1, 1, 1)
        ));
    }

    #[test]
    fn test_deform_conv_config_with_offset_groups() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            2,
            false,
            false,
        )
        .build_with_graph_data(19);
        let processor = DeformConvProcessor;
        let config = processor.extract_config(&node, 19).unwrap();

        assert_eq!(config.offset_groups, 2);
    }

    #[test]
    fn test_deform_conv_config_kernel_shape_inferred() {
        let node = create_test_node(
            vec![],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            1,
            false,
            false,
        )
        .build_with_graph_data(19);
        let processor = DeformConvProcessor;
        let config = processor.extract_config(&node, 19).unwrap();

        assert_eq!(config.kernel_size, [2, 2]); // Inferred from weight shape
    }

    #[test]
    fn test_deform_conv_static_shape_known() {
        // Input [1, 2, 8, 8], weight [4, 2, 2, 2], stride=[1,1], pad=0, dilation=[1,1]
        // H_out = (8 + 0 - 1*(2-1) - 1) / 1 + 1 = 7
        let mut node = TestNodeBuilder::new(NodeType::DeformConv, "test")
            .input_tensor_f32("data", 4, Some(vec![1, 2, 8, 8]))
            .input_tensor_f32_data("weight", vec![0.0; 32], vec![4, 2, 2, 2])
            .input_tensor_f32("offset", 4, None)
            .add_input("", ArgType::default()) // no bias
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", vec![2, 2])
            .attr_ints("strides", vec![1, 1])
            .attr_ints("pads", vec![0, 0, 0, 0])
            .attr_ints("dilations", vec![1, 1])
            .attr_int("group", 1)
            .attr_int("offset_group", 1)
            .build_with_graph_data(19);

        let processor = DeformConvProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 19, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                assert_eq!(
                    t.static_shape,
                    Some(vec![Some(1), Some(4), Some(7), Some(7)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_deform_conv_static_shape_non_default_stride_dilation() {
        // Input [1, 2, 8, 8], weight [4, 2, 3, 3], stride=[2,2], pad=[1,1,1,1], dilation=[2,2]
        // effective_kernel = dilation * (kernel - 1) + 1 = 2*(3-1)+1 = 5
        // H_out = (8 + 2 - 2*(3-1) - 1) / 2 + 1 = (8 + 2 - 5) / 2 + 1 = 5/2 + 1 = 3
        let mut node = TestNodeBuilder::new(NodeType::DeformConv, "test")
            .input_tensor_f32("data", 4, Some(vec![1, 2, 8, 8]))
            .input_tensor_f32_data("weight", vec![0.0; 72], vec![4, 2, 3, 3])
            .input_tensor_f32("offset", 4, None)
            .add_input("", ArgType::default()) // no bias
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", vec![3, 3])
            .attr_ints("strides", vec![2, 2])
            .attr_ints("pads", vec![1, 1, 1, 1])
            .attr_ints("dilations", vec![2, 2])
            .attr_int("group", 1)
            .attr_int("offset_group", 1)
            .build_with_graph_data(19);

        let processor = DeformConvProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 19, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                assert_eq!(
                    t.static_shape,
                    Some(vec![Some(1), Some(4), Some(3), Some(3)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_deform_conv_wrong_input_rank() {
        let mut node = TestNodeBuilder::new(NodeType::DeformConv, "test")
            .input_tensor_f32("data", 3, None) // rank 3 instead of 4
            .input_tensor_f32_data("weight", vec![0.0; 32], vec![4, 2, 2, 2])
            .input_tensor_f32("offset", 4, None)
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", vec![2, 2])
            .attr_ints("strides", vec![1, 1])
            .attr_ints("pads", vec![0, 0, 0, 0])
            .attr_ints("dilations", vec![1, 1])
            .attr_int("group", 1)
            .attr_int("offset_group", 1)
            .build_with_graph_data(19);

        let processor = DeformConvProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 19, &prefs).unwrap_err();
        assert!(err.to_string().contains("rank"), "error: {err}");
    }

    #[test]
    fn test_deform_conv_dtype_mismatch() {
        use crate::ir::DType;
        let mut node = TestNodeBuilder::new(NodeType::DeformConv, "test")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data("weight", vec![0.0; 32], vec![4, 2, 2, 2])
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", vec![2, 2])
            .attr_ints("strides", vec![1, 1])
            .attr_ints("pads", vec![0, 0, 0, 0])
            .attr_ints("dilations", vec![1, 1])
            .attr_int("group", 1)
            .attr_int("offset_group", 1);

        // Add offset with F64 dtype (mismatched)
        node = node.add_input(
            "offset",
            ArgType::Tensor(TensorType {
                dtype: DType::F64,
                rank: 4,
                static_shape: None,
            }),
        );

        let mut node = node.build_with_graph_data(19);
        let processor = DeformConvProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 19, &prefs).unwrap_err();
        assert!(err.to_string().contains("dtype"), "error: {err}");
    }

    #[test]
    fn test_deform_conv_static_shape_no_input_shape() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            1,
            false,
            false,
        )
        .build_with_graph_data(19);
        let mut node = node;
        let processor = DeformConvProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 19, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                assert_eq!(t.static_shape, Some(vec![None, Some(4), None, None]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
