//! Integer convolution and quantized convolution, preserving ONNX geometry attributes.
use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorType};
use crate::node::padding::AutoPad;
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use onnx_ir_derive::NodeBuilder;
/// Original convolution attributes shared by ConvInteger and QLinearConv.
#[derive(Debug, Clone)]
pub struct QuantizedConvConfig {
    /// Automatic padding policy.
    pub auto_pad: AutoPad,
    /// Explicit begin and end padding.
    pub pads: Option<Vec<i64>>,
    /// Optional kernel shape; otherwise taken from the weight tensor.
    pub kernel_shape: Option<Vec<i64>>,
    /// Strides, defaulting to one along every spatial axis.
    pub strides: Option<Vec<i64>>,
    /// Dilations, defaulting to one along every spatial axis.
    pub dilations: Option<Vec<i64>>,
    /// Channel groups, default one.
    pub group: i64,
}
impl Default for QuantizedConvConfig {
    fn default() -> Self {
        Self {
            auto_pad: AutoPad::NotSet,
            pads: None,
            kernel_shape: None,
            strides: None,
            dilations: None,
            group: 1,
        }
    }
}
/// Integer convolution with int32 accumulation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct ConvIntegerNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: QuantizedConvConfig,
}
/// Quantized convolution with int32 accumulation followed by requantization.
#[derive(Debug, Clone, NodeBuilder)]
pub struct QLinearConvNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: QuantizedConvConfig,
}

fn config(node: &RawNode) -> Result<QuantizedConvConfig, ProcessError> {
    let mut c = QuantizedConvConfig::default();
    for (key, value) in &node.attrs {
        match key.as_str() {
            "auto_pad" => c.auto_pad = AutoPad::parse(&value.clone().into_string()?)?,
            "pads" => c.pads = Some(value.clone().into_i64s()?),
            "kernel_shape" => c.kernel_shape = Some(value.clone().into_i64s()?),
            "strides" => c.strides = Some(value.clone().into_i64s()?),
            "dilations" => c.dilations = Some(value.clone().into_i64s()?),
            "group" => c.group = value.clone().into_i64()?,
            _ => {}
        }
    }
    Ok(c)
}
fn infer(node: &mut RawNode, quantized: bool) -> Result<(), ProcessError> {
    let c = config(node)?;
    let wi = if quantized { 3 } else { 1 };
    let (ArgType::Tensor(x), ArgType::Tensor(w)) = (&node.inputs[0].ty, &node.inputs[wi].ty) else {
        return Err(ProcessError::Custom(
            "Quantized convolution requires input and weight tensors".into(),
        ));
    };
    if x.rank < 3 || x.rank != w.rank {
        return Err(ProcessError::Custom(
            "Quantized convolution requires matching ranks of at least 3".into(),
        ));
    }
    for t in [x, w] {
        if !matches!(t.dtype, DType::U8 | DType::I8) {
            return Err(ProcessError::Custom(
                "Quantized convolution requires int8 or uint8 operands".into(),
            ));
        }
    }
    let n = x.rank - 2;
    if w.static_shape
        .as_ref()
        .is_some_and(|shape| shape[2..].contains(&Some(0)))
    {
        return Err(ProcessError::Custom(
            "Convolution kernel dimensions must be positive".into(),
        ));
    }
    if c.auto_pad != AutoPad::NotSet && c.pads.is_some() {
        return Err(ProcessError::Custom(
            "pads and auto_pad cannot be specified together".into(),
        ));
    }
    for (values, len, minimum) in [
        (&c.pads, n * 2, 0),
        (&c.kernel_shape, n, 1),
        (&c.strides, n, 1),
        (&c.dilations, n, 1),
    ] {
        if let Some(values) = values
            && (values.len() != len || values.iter().any(|v| *v < minimum))
        {
            return Err(ProcessError::Custom(
                "Invalid quantized convolution geometry attribute".into(),
            ));
        }
    }
    if c.group <= 0 {
        return Err(ProcessError::Custom(
            "Convolution group must be positive".into(),
        ));
    }
    for (zi, dtype) in [(2, x.dtype), (if quantized { 5 } else { 3 }, w.dtype)] {
        if let Some(z) = node.get_input(zi)
            && (z.ty.elem_type() != dtype || z.ty.rank() > 1)
        {
            return Err(ProcessError::Custom(
                "Convolution zero point must match the operand dtype and be scalar or 1D".into(),
            ));
        }
    }
    if quantized {
        for i in [1, 4, 6] {
            if node.inputs[i].ty.elem_type() != DType::F32 || node.inputs[i].ty.rank() > 1 {
                return Err(ProcessError::Custom(
                    "QLinearConv scale must be float32 and scalar or 1D".into(),
                ));
            }
        }
        if !matches!(node.inputs[7].ty.elem_type(), DType::I8 | DType::U8) {
            return Err(ProcessError::Custom(
                "QLinearConv output zero point must be int8 or uint8".into(),
            ));
        }
        if let Some(bias) = node.get_input(8)
            && (bias.ty.elem_type() != DType::I32 || bias.ty.rank() != 1)
        {
            return Err(ProcessError::Custom(
                "QLinearConv bias must be a 1D int32 tensor".into(),
            ));
        }
    }
    let parameter_len = |index: usize| {
        node.get_input(index).and_then(|arg| match &arg.ty {
            ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => Some(1),
            ArgType::Tensor(t) if t.rank == 1 => t.static_shape.as_ref().and_then(|shape| shape[0]),
            _ => None,
        })
    };
    let output_channels = w.static_shape.as_ref().and_then(|shape| shape[0]);
    for index in if quantized { vec![1, 2, 6, 7] } else { vec![2] } {
        if node.get_input(index).is_some_and(|arg| arg.ty.rank() > 1)
            || parameter_len(index).is_some_and(|len| len != 1)
        {
            return Err(ProcessError::Custom(
                "Convolution input/output quantization parameters must be scalar".into(),
            ));
        }
    }
    for index in if quantized { vec![4, 5] } else { vec![3] } {
        if let (Some(len), Some(channels)) = (parameter_len(index), output_channels)
            && len != 1
            && len != channels
        {
            return Err(ProcessError::Custom(
                "Weight quantization parameters must be scalar or match output channels".into(),
            ));
        }
    }
    if quantized
        && let (Some(len), Some(channels)) = (parameter_len(8), output_channels)
        && len != channels
    {
        return Err(ProcessError::Custom(
            "Convolution bias length must match output channels".into(),
        ));
    }
    let dtype = if quantized {
        node.inputs[7].ty.elem_type()
    } else {
        DType::I32
    };
    let dim = |t: &TensorType, axis: usize| t.static_shape.as_ref().and_then(|shape| shape[axis]);
    let groups = c.group as usize;
    if let Some(channels) = dim(x, 1)
        && (channels % groups != 0 || dim(w, 1).is_some_and(|wc| wc * groups != channels))
    {
        return Err(ProcessError::Custom(
            "Convolution channels must match the weight groups".into(),
        ));
    }
    if dim(w, 0).is_some_and(|out| out % groups != 0) {
        return Err(ProcessError::Custom(
            "Output channels must divide into groups".into(),
        ));
    }
    let mut shape = vec![dim(x, 0), dim(w, 0)];
    for axis in 0..n {
        let kernel = c
            .kernel_shape
            .as_ref()
            .map(|v| v[axis] as usize)
            .or_else(|| dim(w, axis + 2));
        if let (Some(declared), Some(actual)) = (
            c.kernel_shape.as_ref().map(|v| v[axis] as usize),
            dim(w, axis + 2),
        ) && declared != actual
        {
            return Err(ProcessError::Custom(
                "kernel_shape must match the weight dimensions".into(),
            ));
        }
        let stride = c.strides.as_ref().map_or(1, |v| v[axis] as usize);
        let dilation = c.dilations.as_ref().map_or(1, |v| v[axis] as usize);
        let output = match (dim(x, axis + 2), kernel) {
            (Some(size), _) if matches!(c.auto_pad, AutoPad::SameUpper | AutoPad::SameLower) => {
                Some(size.div_ceil(stride))
            }
            (Some(size), Some(kernel)) => {
                let pad = if c.auto_pad == AutoPad::NotSet {
                    c.pads
                        .as_ref()
                        .map_or(0, |v| (v[axis] + v[axis + n]) as usize)
                } else {
                    0
                };
                size.checked_add(pad)
                    .and_then(|size| size.checked_sub((kernel - 1) * dilation + 1))
                    .map(|size| size / stride + 1)
            }
            _ => None,
        };
        shape.push(output);
    }
    node.outputs[0].ty = ArgType::Tensor(TensorType::new(dtype, x.rank, Some(shape)));
    Ok(())
}
macro_rules! processor {
    ($processor:ident, $node:ident, $variant:ident, $quantized:literal, $min:literal, $max:literal) => {
        pub(crate) struct $processor;
        impl NodeProcessor for $processor {
            type Config = QuantizedConvConfig;
            fn spec(&self) -> NodeSpec {
                NodeSpec {
                    min_opset: 10,
                    max_opset: None,
                    inputs: InputSpec::Range($min, $max),
                    outputs: OutputSpec::Exact(1),
                }
            }
            fn extract_config(
                &self,
                node: &RawNode,
                _: usize,
            ) -> Result<Self::Config, ProcessError> {
                config(node)
            }
            fn infer_types(
                &self,
                node: &mut RawNode,
                _: usize,
                _: &OutputPreferences,
            ) -> Result<(), ProcessError> {
                infer(node, $quantized)
            }
            fn build_node(&self, node: RawNode, _: usize) -> Result<Node, ProcessError> {
                let config = config(&node)?;
                Ok(Node::$variant($node {
                    name: node.name,
                    inputs: node.inputs,
                    outputs: node.outputs,
                    config,
                }))
            }
        }
    };
}
processor!(
    ConvIntegerProcessor,
    ConvIntegerNode,
    ConvInteger,
    false,
    2,
    4
);
processor!(
    QLinearConvProcessor,
    QLinearConvNode,
    QLinearConv,
    true,
    8,
    9
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    #[test]
    fn shape_and_geometry_validation() {
        let mut node = TestNodeBuilder::new(NodeType::ConvInteger, "conv")
            .input_tensor_i8("x", 4, Some(vec![1, 4, 4, 4]))
            .input_tensor_i8("w", 4, Some(vec![4, 2, 2, 2]))
            .output_default("y")
            .attr_int("group", 2)
            .attr_ints("strides", vec![2, 2])
            .attr_ints("dilations", vec![2, 2])
            .attr_ints("pads", vec![1, 1, 0, 0])
            .build();
        ConvIntegerProcessor
            .infer_types(&mut node, 10, &OutputPreferences::default())
            .unwrap();
        assert_eq!(
            node.outputs[0].ty,
            ArgType::Tensor(TensorType::new(
                DType::I32,
                4,
                Some(vec![Some(1), Some(4), Some(2), Some(2)])
            ))
        );
        node.attrs
            .insert("group".into(), crate::ir::AttributeValue::Int64(3));
        assert!(
            ConvIntegerProcessor
                .infer_types(&mut node, 10, &OutputPreferences::default())
                .is_err()
        );
    }
}
