//! # LpPool (1D)
//!
//! 1D Lp pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LpPool.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version. `p` is a FLOAT attribute.
//! - **Opset 2**: `p` becomes an INT attribute.
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::node::global_lp_pool::extract_p;
use crate::node::padding::padding_config_1d;
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use super::padding::{AutoPad, PaddingConfig1d};

/// Configuration for LpPool1d operations extracted from ONNX nodes
#[derive(Debug, Clone, new)]
pub struct LpPool1dConfig {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding configuration
    pub padding: PaddingConfig1d,
    /// Dilation (opset 11+)
    pub dilation: usize,
    /// Whether to use ceil mode for output size calculation (opset 18+)
    pub ceil_mode: bool,
    /// Auto padding mode
    pub auto_pad: AutoPad,
    /// Norm type p (defaults to 2). Held as a float because opset 1 declares `p` as
    /// FLOAT and does not restrict it to whole numbers.
    pub p: f64,
}

/// Node representation for LpPool1d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct LpPool1dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LpPool1dConfig,
}

pub(crate) struct LpPool1dProcessor;

impl NodeProcessor for LpPool1dProcessor {
    type Config = LpPool1dConfig;

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
        let mut has_kernel_shape = false;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => {
                    has_kernel_shape = true;
                    let kernel_shape = value.clone().into_i64s();
                    if kernel_shape.len() != 1 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool1d: kernel_shape must have length 1, got {:?}",
                            kernel_shape
                        )));
                    }
                    if kernel_shape[0] <= 0 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool1d: kernel_shape values must be > 0, got {:?}",
                            kernel_shape
                        )));
                    }
                }
                "strides" => {
                    let strides = value.clone().into_i64s();
                    if strides.len() != 1 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool1d: strides must have length 1, got {:?}",
                            strides
                        )));
                    }
                    if strides[0] <= 0 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool1d: strides values must be > 0, got {:?}",
                            strides
                        )));
                    }
                }
                "pads" => {
                    let pads = value.clone().into_i64s();
                    if pads.len() != 2 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool1d: pads must have length 2, got {:?}",
                            pads
                        )));
                    }
                }
                "p" => {
                    extract_p(node)?;
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
                    if dilations.len() != 1 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool1d: dilations must have length 1, got {:?}",
                            dilations
                        )));
                    }
                    if dilations[0] <= 0 {
                        return Err(ProcessError::Custom(format!(
                            "LpPool1d: dilations values must be > 0, got {:?}",
                            dilations
                        )));
                    }
                    if dilations.iter().any(|&d| d != 1) {
                        if opset < 11 {
                            return Err(ProcessError::Custom(format!(
                                "LpPool1d: dilations requires opset 11+, got opset {}",
                                opset
                            )));
                        }
                        return Err(ProcessError::Custom(
                            "LpPool1d: dilations != 1 is not supported in burn-onnx yet"
                                .to_string(),
                        ));
                    }
                }
                "auto_pad" => {
                    AutoPad::parse(&value.clone().into_string())?;
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for LpPool1d: {key}"),
                    });
                }
            }
        }

        if !has_kernel_shape {
            return Err(ProcessError::Custom(
                "LpPool1d: missing required attribute kernel_shape".to_string(),
            ));
        }

        crate::node::padding::validate_auto_pad(node)?;

        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut stride = vec![1];
        let mut pads = vec![0, 0];
        let mut dilations = vec![1];
        let mut ceil_mode: i64 = 0;
        let mut auto_pad = AutoPad::NotSet;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => stride = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "ceil_mode" => ceil_mode = value.clone().into_i64(),
                "auto_pad" => auto_pad = AutoPad::parse(&value.clone().into_string())?,
                _ => {}
            }
        }

        let p = extract_p(node)?;
        let padding = padding_config_1d(&pads);

        let config = LpPool1dConfig::new(
            kernel_shape[0] as usize,
            stride[0] as usize,
            padding,
            dilations[0] as usize,
            ceil_mode != 0,
            auto_pad,
            p,
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::LpPool1d(LpPool1dNode {
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
        let mut builder = TestNodeBuilder::new(NodeType::LpPool1d, "test_lppool1d")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
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
    fn test_lppool1d_config_basic() {
        let node = create_test_node(vec![3], vec![1], vec![0, 0], None, 0, None);
        let mut node = node;
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 3);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.p, 2.0);
        assert!(!config.ceil_mode);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_lppool1d_config_with_custom_p() {
        let node = create_test_node(vec![3], vec![2], vec![1, 1], None, 0, Some(3));
        let mut node = node;
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 3);
        assert_eq!(config.stride, 2);
        assert_eq!(config.p, 3.0);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(1, 1)));
    }

    #[test]
    fn test_lppool1d_ceil_mode_opset_validation() {
        let mut node = create_test_node(vec![3], vec![1], vec![0, 0], None, 1, None);
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("ceil_mode requires opset 18+"));
    }

    #[test]
    fn test_lppool1d_missing_kernel_shape_validation() {
        let mut node = TestNodeBuilder::new(NodeType::LpPool1d, "test_lppool1d_missing_kernel")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .build();
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("missing required attribute kernel_shape"));
    }

    #[test]
    fn test_lppool1d_dilation_unsupported_validation() {
        let mut node = create_test_node(vec![3], vec![1], vec![0, 0], Some(vec![2]), 0, None);
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("dilations != 1 is not supported"));
    }

    #[test]
    fn test_lppool1d_non_positive_values_validation() {
        let mut kernel_zero = create_test_node(vec![0], vec![1], vec![0, 0], None, 0, None);
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();
        let err = processor
            .infer_types(&mut kernel_zero, 16, &prefs)
            .expect_err("Expected non-positive kernel_shape to fail");
        assert!(format!("{}", err).contains("kernel_shape values must be > 0"));

        let mut stride_zero = create_test_node(vec![3], vec![0], vec![0, 0], None, 0, None);
        let err = processor
            .infer_types(&mut stride_zero, 16, &prefs)
            .expect_err("Expected non-positive stride to fail");
        assert!(format!("{}", err).contains("strides values must be > 0"));

        let mut dilation_zero =
            create_test_node(vec![3], vec![1], vec![0, 0], Some(vec![0]), 0, None);
        let err = processor
            .infer_types(&mut dilation_zero, 16, &prefs)
            .expect_err("Expected non-positive dilation to fail");
        assert!(format!("{}", err).contains("dilations values must be > 0"));

        let mut p_zero = create_test_node(vec![3], vec![1], vec![0, 0], None, 0, Some(0));
        let err = processor
            .infer_types(&mut p_zero, 16, &prefs)
            .expect_err("Expected non-positive p to fail");
        assert!(format!("{}", err).contains("p must be finite and > 0"));
    }

    #[test]
    fn test_lppool1d_dilation_opset_validation() {
        let mut node = create_test_node(vec![3], vec![1], vec![0, 0], Some(vec![2]), 0, None);
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 10, &prefs);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("dilations requires opset 11+"));
    }

    /// Pins `validate_auto_pad` into this processor's `infer_types`. Without the call the node
    /// reaches burn-onnx codegen, which panics instead of reporting a `ProcessError`.
    #[test]
    fn test_lp_pool1d_rejects_unsupported_auto_pad_on_dynamic_shape() {
        let mut node = TestNodeBuilder::new(NodeType::LpPool1d, "test_auto_pad")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_ints("kernel_shape", vec![3])
            .attr_int("p", 2)
            .attr_string("auto_pad", "SAME_LOWER")
            .build_with_graph_data(18);
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();
        let err = processor
            .infer_types(&mut node, 18, &prefs)
            .expect_err("SAME_LOWER over a dynamic input must be rejected")
            .to_string();
        assert!(
            err.contains(&crate::node::padding::SameBlocker::SameLower.to_string()),
            "{err}"
        );
    }

    /// Opset 1 declares `p` as FLOAT, and it may be fractional.
    #[test]
    fn test_lppool1d_opset1_float_p() {
        let mut node = TestNodeBuilder::new(NodeType::LpPool1d, "test_float_p")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_ints("kernel_shape", vec![3])
            .attr_float("p", 1.5)
            .build();
        let processor = LpPool1dProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 1, &prefs).unwrap();
        let config = processor.extract_config(&node, 1).unwrap();
        assert_eq!(config.p, 1.5);
    }
}
