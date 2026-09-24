//! # MaxPool (1D)
//!
//! 1D max pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MaxPool.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic max pooling operation.
//! - **Opset 8**: Added optional Indices output and the `storage_order` attribute.
//! - **Opset 10**: Added `ceil_mode` attribute to use ceiling instead of floor for output shape calculation.
//! - **Opset 11**: Added support for dilation; updated padding semantics.
//! - **Opset 12**: Added support for int8, uint8 data types; clarified behavior with negative padding.
//!
//! **Implementation Note**: Accepts 1-2 outputs (Y required, optional Indices output).
//! Indices are typed as int64 with the input's rank.
//!
//! ## Missing Test Coverage
//! - TODO: No test for dilation > 1 with opset < 11 - Should reject dilation in older opsets
//! - TODO: No test for int8/uint8 dtypes - Opset 12+ supports integer types
//! - TODO: No test for kernel_shape validation - Missing kernel_shape attribute should be rejected
//! - TODO: No test for negative padding values - Opset 12+ allows negative padding
//! - TODO: No test for edge case: kernel larger than input dimension
//! - TODO: No test validating input is 3D (N x C x L) - Lower/higher rank should be rejected
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::{
    ir::{ArgType, Argument, DType, Node, RawNode, TensorType},
    node::padding::padding_config_1d,
};

use super::padding::{AutoPad, PaddingConfig1d};

/// Configuration for MaxPool1d operations extracted from ONNX nodes
#[derive(Debug, Clone, new)]
pub struct MaxPool1dConfig {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Dilation
    pub dilation: usize,
    /// Padding configuration
    pub padding: PaddingConfig1d,
    /// Whether to use ceil mode for output size calculation (opset 10+)
    pub ceil_mode: bool,
    /// Auto padding mode
    pub auto_pad: AutoPad,
    /// Layout of the optional Indices output: 0 row-major (default), 1 column-major.
    /// Both flatten a single spatial axis the same way.
    #[new(default)]
    pub storage_order: i64,
}

/// Node representation for MaxPool1d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MaxPool1dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: MaxPool1dConfig,
}

impl MaxPool1dConfig {
    /// Set the stride
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding configuration
    pub fn with_padding(mut self, padding: PaddingConfig1d) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }
}

pub(crate) struct MaxPool1dProcessor;

impl NodeProcessor for MaxPool1dProcessor {
    type Config = MaxPool1dConfig;

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
        // TODO: Validate input tensor is 3D (N x C x L) - Lower or higher rank should be rejected
        // TODO: Validate input dtype - int8/uint8 support requires opset 12+

        // Validate attributes before extracting config
        // TODO: Validate required kernel_shape attribute is present - Missing kernel_shape should cause error

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" => {}
                "storage_order" => {}
                "dilations" => {
                    // Dilation support requires opset 11+
                    let dilations = value.clone().into_i64s();
                    if dilations.iter().any(|&d| d != 1) && opset < 11 {
                        return Err(ProcessError::Custom(format!(
                            "MaxPool: dilation requires opset 11+, got opset {}",
                            opset
                        )));
                    }
                }
                "auto_pad" => {
                    AutoPad::parse(&value.clone().into_string())?;
                }
                "ceil_mode" => {
                    // ceil_mode support requires opset 10+
                    let ceil_mode = value.clone().into_i64();
                    if ceil_mode != 0 && opset < 10 {
                        return Err(ProcessError::Custom(format!(
                            "MaxPool: ceil_mode requires opset 10+, got opset {}",
                            opset
                        )));
                    }
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for MaxPool1d: {key}"),
                    });
                }
            }
        }

        crate::node::padding::validate_auto_pad(node)?;

        // Output type is same as input
        crate::processor::same_as_input(node);

        // The optional Indices output holds int64 positions into the flattened input.
        if let Some(indices) = node.outputs.get_mut(1) {
            let rank = node.inputs[0].ty.rank();
            indices.ty = ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank,
                static_shape: None,
            });
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut stride = vec![1];
        let mut pads = vec![0, 0];
        let mut dilation = vec![1];
        let mut ceil_mode: i64 = 0;
        let mut auto_pad = AutoPad::NotSet;
        let mut storage_order = 0;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => stride = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilation = value.clone().into_i64s(),
                "ceil_mode" => ceil_mode = value.clone().into_i64(),
                "auto_pad" => auto_pad = AutoPad::parse(&value.clone().into_string())?,
                "storage_order" => storage_order = value.clone().into_i64(),
                _ => {}
            }
        }

        if !matches!(storage_order, 0 | 1) {
            return Err(ProcessError::InvalidAttribute {
                name: "storage_order".to_string(),
                reason: format!("expected 0 (row major) or 1 (column major), got {storage_order}"),
            });
        }

        let padding = padding_config_1d(&pads);

        let mut config = MaxPool1dConfig::new(
            kernel_shape[0] as usize,
            stride[0] as usize,
            dilation[0] as usize,
            padding,
            ceil_mode == 1,
            auto_pad,
        );
        config.storage_order = storage_order;

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::MaxPool1d(MaxPool1dNode {
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
    use crate::{ir::NodeType, node::padding::PaddingConfig1d, node::test_utils::TestNodeBuilder};

    fn create_test_node(
        kernel_shape: Vec<i64>,
        stride: Vec<i64>,
        pads: Vec<i64>,
        dilation: Vec<i64>,
        ceil_mode: i64,
        auto_pad: Option<&str>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::MaxPool1d, "test_maxpool1d")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", stride)
            .attr_ints("pads", pads)
            .attr_int("ceil_mode", ceil_mode)
            .attr_ints("dilations", dilation);
        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }
        builder.build()
    }

    #[test]
    fn test_max_pool1d_config_basic() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(!config.ceil_mode);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_config_with_padding() {
        let node = create_test_node(vec![4], vec![2], vec![2, 2], vec![1], 0, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2, 2)));
    }

    #[test]
    fn test_max_pool1d_config_with_dilation() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![2], 0, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 2);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_config_asymmetric_padding() {
        let node = create_test_node(vec![4], vec![1], vec![1, 2], vec![1], 0, None);
        let processor = MaxPool1dProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        // Asymmetric padding should now be captured instead of panicking
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(1, 2)));
        assert!(config.padding.is_asymmetric());
    }

    #[test]
    fn test_max_pool1d_config_auto_pad_not_set() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, Some("NOTSET"));
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_config_auto_pad_same_upper() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, Some("SAME_UPPER"));
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.auto_pad, AutoPad::SameUpper);
    }

    #[test]
    fn test_max_pool1d_config_with_ceil_mode() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 1, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(config.ceil_mode);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_ceil_mode_opset_validation() {
        // Test that ceil_mode=1 with opset < 10 is rejected
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 1, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 9, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
        if let Err(ProcessError::Custom(msg)) = result {
            assert!(msg.contains("ceil_mode requires opset 10+"));
        }
    }

    #[test]
    fn test_max_pool1d_indices_output_is_i64() {
        // Declared as f32 so the test shows infer_types retypes it.
        let mut node = TestNodeBuilder::new(NodeType::MaxPool1d, "test_indices")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .output_tensor_f32("indices", 3, None)
            .attr_ints("kernel_shape", vec![2])
            .attr_int("storage_order", 1)
            .build();
        let processor = MaxPool1dProcessor;
        processor
            .infer_types(&mut node, 12, &OutputPreferences::new())
            .unwrap();
        assert!(matches!(
            node.outputs[1].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 3,
                ..
            })
        ));
        let config = processor.extract_config(&node, 12).unwrap();
        assert_eq!(config.storage_order, 1);
    }

    #[test]
    fn test_max_pool1d_rejects_unknown_storage_order() {
        let node = TestNodeBuilder::new(NodeType::MaxPool1d, "test_storage_order")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_ints("kernel_shape", vec![2])
            .attr_int("storage_order", 2)
            .build();
        let result = MaxPool1dProcessor.extract_config(&node, 16);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "storage_order"
        ));
    }

    #[test]
    fn test_max_pool1d_ceil_mode_zero_accepted_old_opset() {
        // Test that ceil_mode=0 is accepted even with old opset
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 1, &prefs);
        assert!(result.is_ok());
    }

    /// Pins `validate_auto_pad` into this processor's `infer_types`. Without the call the node
    /// reaches burn-onnx codegen, which panics instead of reporting a `ProcessError`.
    #[test]
    fn test_max_pool1d_rejects_unsupported_auto_pad_on_dynamic_shape() {
        let mut node = TestNodeBuilder::new(NodeType::MaxPool1d, "test_auto_pad")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_ints("kernel_shape", vec![3])
            .attr_string("auto_pad", "SAME_LOWER")
            .build_with_graph_data(16);
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let err = processor
            .infer_types(&mut node, 16, &prefs)
            .expect_err("SAME_LOWER over a dynamic input must be rejected")
            .to_string();
        assert!(
            err.contains(&crate::node::padding::SameBlocker::SameLower.to_string()),
            "{err}"
        );
    }
}
