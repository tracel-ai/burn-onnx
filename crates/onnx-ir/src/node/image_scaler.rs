//! Legacy experimental ImageScaler (opset 1): scale an NCHW image and add channel biases.
use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use onnx_ir_derive::NodeBuilder;

/// Original ONNX ImageScaler attributes.
#[derive(Debug, Clone)]
pub struct ImageScalerConfig {
    /// Multiplier applied to every pixel (default 1).
    pub scale: f32,
    /// Optional bias for each channel; an absent list means no bias.
    pub bias: Vec<f32>,
}
impl Default for ImageScalerConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            bias: Vec::new(),
        }
    }
}
/// Scale and bias an image in NCHW layout.
#[derive(Debug, Clone, NodeBuilder)]
pub struct ImageScalerNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ImageScalerConfig,
}
pub(crate) struct ImageScalerProcessor;
impl NodeProcessor for ImageScalerProcessor {
    type Config = ImageScalerConfig;
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }
    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut config = ImageScalerConfig::default();
        if let Some(value) = node.attrs.get("scale") {
            config.scale = value.clone().into_f32()?;
        }
        if let Some(value) = node.attrs.get("bias") {
            config.bias = value.clone().into_f32s()?;
        }
        Ok(config)
    }
    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let config = self.extract_config(node, opset)?;
        let ArgType::Tensor(input) = &node.inputs[0].ty else {
            return Err(ProcessError::Custom(
                "ImageScaler requires a floating-point NCHW tensor".into(),
            ));
        };
        if input.rank != 4 || !input.dtype.is_float() {
            return Err(ProcessError::Custom(
                "ImageScaler requires a floating-point NCHW tensor".into(),
            ));
        }
        if !config.bias.is_empty()
            && let Some(channels) = input.static_shape.as_ref().and_then(|s| s[1])
            && channels != config.bias.len()
        {
            return Err(ProcessError::Custom(format!(
                "ImageScaler bias length {} must match channel count {channels}",
                config.bias.len()
            )));
        }
        crate::processor::same_as_input(node);
        Ok(())
    }
    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;
        Ok(Node::ImageScaler(ImageScalerNode {
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
    #[test]
    fn defaults_and_validation() {
        let mut node = TestNodeBuilder::new(NodeType::ImageScaler, "scale")
            .input_tensor_f32("x", 4, Some(vec![2, 3, 4, 5]))
            .output_default("y")
            .build();
        let config = ImageScalerProcessor.extract_config(&node, 1).unwrap();
        assert_eq!(config.scale, 1.0);
        assert!(config.bias.is_empty());
        ImageScalerProcessor
            .infer_types(&mut node, 1, &OutputPreferences::default())
            .unwrap();
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
        let mut invalid = TestNodeBuilder::new(NodeType::ImageScaler, "scale")
            .input_tensor_f32("x", 3, None)
            .output_default("y")
            .build();
        assert!(
            ImageScalerProcessor
                .infer_types(&mut invalid, 1, &OutputPreferences::default())
                .is_err()
        );
    }
    #[test]
    fn attributes_and_channel_validation() {
        let mut node = TestNodeBuilder::new(NodeType::ImageScaler, "scale")
            .input_tensor_f32("x", 4, Some(vec![2, 3, 4, 5]))
            .output_default("y")
            .attr_float("scale", 0.5)
            .attr_floats("bias", vec![1.0, -2.0, 3.0])
            .attr_int("future_attribute", 1)
            .build();
        let config = ImageScalerProcessor.extract_config(&node, 1).unwrap();
        assert_eq!(config.scale, 0.5);
        assert_eq!(config.bias, vec![1.0, -2.0, 3.0]);
        ImageScalerProcessor
            .infer_types(&mut node, 1, &OutputPreferences::default())
            .unwrap();
        let mut invalid = TestNodeBuilder::new(NodeType::ImageScaler, "scale")
            .input_tensor_f32("x", 4, Some(vec![2, 3, 4, 5]))
            .output_default("y")
            .attr_floats("bias", vec![1.0])
            .build();
        assert!(
            ImageScalerProcessor
                .infer_types(&mut invalid, 1, &OutputPreferences::default())
                .unwrap_err()
                .to_string()
                .contains("channel count")
        );
    }
}
