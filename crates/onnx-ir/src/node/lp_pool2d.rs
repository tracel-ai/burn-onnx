//! # LpPool (2D)
//!
//! 2D Lp pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LpPool.html>
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::node::padding::{AutoPad, PaddingConfig2d, padding_config_2d};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for LpPool2d operations
#[derive(Debug, Clone, new)]
pub struct LpPool2dConfig {
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub strides: [usize; 2],
    /// Padding configuration
    pub padding: PaddingConfig2d,
    /// Dilation [height, width] (opset 11+)
    pub dilation: [usize; 2],
    /// Whether to use ceil mode for output size calculation (opset 18+)
    pub ceil_mode: bool,
    /// Auto padding mode
    pub auto_pad: AutoPad,
    /// Norm type p (defaults to 2)
    pub p: i64,
}

/// Node representation for LpPool2d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct LpPool2dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LpPool2dConfig,
}

pub(crate) struct LpPool2dProcessor;

impl NodeProcessor for LpPool2dProcessor {
    type Config = LpPool2dConfig;

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
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" => {}
                "p" => {
                    let p = value.clone().into_i64();
                    if p <= 0 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool: p must be > 0, got {}",
                            p
                        )));
                    }
                }
                "ceil_mode" => {
                    let ceil_mode = value.clone().into_i64();
                    if ceil_mode != 0 && opset < 18 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool: ceil_mode requires opset 18+, got opset {}",
                            opset
                        )));
                    }
                }
                "dilations" => {
                    let dilations = value.clone().into_i64s();
                    if dilations.iter().any(|&d| d != 1) && opset < 11 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool: dilations requires opset 11+, got opset {}",
                            opset
                        )));
                    }
                }
                "auto_pad" => {
                    AutoPad::parse(&value.clone().into_string())?;
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for LpPool2d: {key}"),
                    });
                }
            }
        }

        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1];
        let mut pads = vec![0, 0, 0, 0];
        let mut dilations = vec![1, 1];
        let mut ceil_mode: i64 = 0;
        let mut auto_pad = AutoPad::NotSet;
        let mut p: i64 = 2;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "ceil_mode" => ceil_mode = value.clone().into_i64(),
                "auto_pad" => auto_pad = AutoPad::parse(&value.clone().into_string())?,
                "p" => p = value.clone().into_i64(),
                _ => {}
            }
        }

        let padding = padding_config_2d(&pads);

        let config = LpPool2dConfig::new(
            [kernel_shape[0] as usize, kernel_shape[1] as usize],
            [strides[0] as usize, strides[1] as usize],
            padding,
            [dilations[0] as usize, dilations[1] as usize],
            ceil_mode == 1,
            auto_pad,
            p,
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::LpPool2d(LpPool2dNode {
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
        dilations: Option<Vec<i64>>,
        ceil_mode: i64,
        p: Option<i64>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::LpPool2d, "test_lppool2d")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_int("ceil_mode", ceil_mode);

        if let Some(dilations) = dilations {
            builder = builder.attr_ints("dilations", dilations);
        }

        if let Some(p) = p {
            builder = builder.attr_int("p", p);
        }

        builder.build()
    }

    #[test]
    fn test_lppool2d_config_basic() {
        let node = create_test_node(vec![2, 3], vec![1, 2], vec![0, 1, 1, 0], None, 0, None);
        let mut node = node;
        let processor = LpPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [2, 3]);
        assert_eq!(config.strides, [1, 2]);
        assert_eq!(config.dilation, [1, 1]);
        assert_eq!(config.p, 2);
        assert!(!config.ceil_mode);
        assert!(matches!(
            config.padding,
            PaddingConfig2d::Explicit(0, 1, 1, 0)
        ));
    }

    #[test]
    fn test_lppool2d_config_with_custom_p() {
        let node = create_test_node(vec![3, 3], vec![2, 2], vec![1, 1, 1, 1], None, 0, Some(4));
        let mut node = node;
        let processor = LpPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [2, 2]);
        assert_eq!(config.p, 4);
    }

    #[test]
    fn test_lppool2d_ceil_mode_opset_validation() {
        let mut node = create_test_node(vec![2, 2], vec![1, 1], vec![0, 0, 0, 0], None, 1, None);
        let processor = LpPool2dProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("ceil_mode requires opset 18+"));
    }
}
