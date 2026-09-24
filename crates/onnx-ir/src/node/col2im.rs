//! # Col2Im
//!
//! Rearranges column blocks back into a multidimensional image.
//! This is the reverse operation of Im2Col.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Col2Im.html>
//!
//! ## Opset Versions
//! - **Opset 18**: Initial version
//!
//! ## Extensions
//! - **1D Support**: The ONNX specification requires `image_shape` and `block_shape` to be at least 2D.
//!   This implementation extends support to 1D `image_shape` and `block_shape` as well.
//!
//! ## Inputs
//! - `data` (tensor(float32/float16/bfloat16)): Input tensor of shape `[N, C * prod(block_shape), L]`
//! - `image_shape` (tensor(int64)): The shape of the spatial dimensions of the image
//! - `block_shape` (tensor(int64)): The shape of the block to apply on the image
//!
//! ## Attributes
//! - `dilations` (list of ints, default all 1s): Dilation value along each spatial axis
//! - `pads` (list of ints, default all 0s): Padding for the beginning and ending along each spatial axis
//! - `strides` (list of ints, default all 1s): Stride along each spatial axis

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, RuntimeInputRef, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for Col2Im operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct Col2ImNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: Col2ImConfig,
}

/// A Col2Im shape input: known at build time, or read from the input at run time.
#[derive(Debug, Clone, PartialEq)]
pub enum Col2ImShape {
    /// Values known at build time.
    Static(Vec<usize>),
    /// Values read from a runtime input holding one entry per spatial axis.
    Runtime {
        /// The input holding the values.
        input: RuntimeInputRef,
        /// Number of spatial axes, which is the input's length.
        len: usize,
    },
}

impl Col2ImShape {
    /// Number of spatial axes this shape describes.
    pub fn len(&self) -> usize {
        match self {
            Col2ImShape::Static(values) => values.len(),
            Col2ImShape::Runtime { len, .. } => *len,
        }
    }

    /// Whether the shape describes no spatial axes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The build-time values, if known.
    pub fn as_static(&self) -> Option<&[usize]> {
        match self {
            Col2ImShape::Static(values) => Some(values),
            Col2ImShape::Runtime { .. } => None,
        }
    }
}

/// Configuration for Col2Im operation
#[derive(Debug, Clone, new)]
pub struct Col2ImConfig {
    /// Image shape (spatial dimensions of the output image)
    pub image_shape: Col2ImShape,
    /// Block shape (kernel size)
    pub block_shape: Col2ImShape,
    /// Dilation value along each spatial axis
    pub dilations: Vec<usize>,
    /// Padding for the beginning and ending along each spatial axis
    /// Format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    pub pads: Vec<usize>,
    /// Stride along each spatial axis
    pub strides: Vec<usize>,
}

pub(crate) struct Col2ImProcessor;

impl NodeProcessor for Col2ImProcessor {
    type Config = Col2ImConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 18,
            max_opset: None,
            inputs: InputSpec::Exact(3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift image_shape (input[1]) if constant
        if node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        // Lift block_shape (input[2]) if constant
        if node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }
        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate attributes

        // Validate data input is a tensor
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{}", node.inputs[0].ty),
                });
            }
        };

        // Col2Im data input should be rank 3: [N, C * product(block_shape), L]
        if tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "Col2Im expects data input tensor of rank 3 (N x C*prod(block_shape) x L), got rank {}",
                tensor.rank
            )));
        }

        // Extract config to get image_shape and block_shape
        let config = self.extract_config(node, opset)?;

        // Output rank: batch + channels + spatial dimensions
        // Output shape: [N, C, *image_shape]
        // where C = input_shape[1] / product(block_shape)
        let num_spatial_dims = config.image_shape.len();
        let output_rank = 2 + num_spatial_dims; // N + C + spatial dims

        // Use partial static shape inference:
        // Always attempt to compute static_shape if config is available, even if input is dynamic.
        // We know (N, C, *image_shape) structure.
        let (n, c) = match &tensor.static_shape {
            Some(input_shape) => {
                let c = config.block_shape.as_static().and_then(|block| {
                    let block_product: usize = block.iter().product();
                    input_shape[1].map(|v| v / block_product)
                });
                (input_shape[0], c)
            }
            None => (None, None),
        };
        let spatial: Vec<Option<usize>> = match config.image_shape.as_static() {
            Some(image) => image.iter().map(|&dim| Some(dim)).collect(),
            None => vec![None; num_spatial_dims],
        };
        let static_shape = Some([vec![n, c], spatial].concat());

        // Validate supported dimensions (only 1D and 2D supported by current codegen)
        if num_spatial_dims > 2 {
            return Err(ProcessError::Custom(format!(
                "Col2Im currently only supports 1D and 2D spatial dimensions, got {}",
                num_spatial_dims
            )));
        }

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: output_rank,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let image_shape = shape_input(node, 1, "image_shape")?;
        let block_shape = shape_input(node, 2, "block_shape")?;
        if image_shape.len() != block_shape.len() {
            return Err(ProcessError::Custom(format!(
                "Col2Im: image_shape has {} entries but block_shape has {}",
                image_shape.len(),
                block_shape.len()
            )));
        }

        let num_spatial_dims = image_shape.len();

        // Note: ONNX spec requires num_spatial_dims >= 2, but we support 1D as an extension.

        if num_spatial_dims == 0 {
            return Err(ProcessError::Custom(
                "Col2Im: image_shape and block_shape must not be empty".to_string(),
            ));
        }

        // pads holds the begin values, then the end values.
        let dilations = int_list_attr(node, "dilations", num_spatial_dims, 1)?;
        let pads = int_list_attr(node, "pads", num_spatial_dims * 2, 0)?;
        let strides = int_list_attr(node, "strides", num_spatial_dims, 1)?;

        Ok(Col2ImConfig::new(
            image_shape,
            block_shape,
            dilations,
            pads,
            strides,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Col2Im(Col2ImNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

/// Read a list attribute of `len` entries. Its default is also its minimum: 1 for
/// dilations and strides, 0 for pads.
fn int_list_attr(
    node: &RawNode,
    name: &str,
    len: usize,
    least: i64,
) -> Result<Vec<usize>, ProcessError> {
    let Some(value) = node.attrs.get(name) else {
        return Ok(vec![least as usize; len]);
    };
    let values = value.clone().into_i64s();
    if values.len() != len {
        return Err(ProcessError::InvalidAttribute {
            name: name.to_string(),
            reason: format!("expected {len} entries, got {}", values.len()),
        });
    }
    if let Some(&v) = values.iter().find(|&&v| v < least) {
        return Err(ProcessError::InvalidAttribute {
            name: name.to_string(),
            reason: format!("entries must be at least {least}, got {v}"),
        });
    }
    Ok(values.iter().map(|&v| v as usize).collect())
}

/// Read a Col2Im shape input: its value when constant, otherwise a reference to it
/// with the spatial rank taken from its known length.
fn shape_input(node: &RawNode, index: usize, name: &str) -> Result<Col2ImShape, ProcessError> {
    use crate::ir::TensorDataExt;

    let arg = &node.inputs[index];
    if let Some(data) = arg.value() {
        let values = data
            .to_i64_vec()
            .map_err(|_| ProcessError::Custom(format!("Col2Im: {name} must be an int64 tensor")))?;
        if let Some(&v) = values.iter().find(|&&v| v <= 0) {
            return Err(ProcessError::Custom(format!(
                "Col2Im: {name} entries must be positive, got {v}"
            )));
        }
        return Ok(Col2ImShape::Static(
            values.iter().map(|&v| v as usize).collect(),
        ));
    }

    let len = match &arg.ty {
        ArgType::Tensor(t) if t.rank != 1 || !t.dtype.is_int() => None,
        ty => ty.first_dim_static_len(),
    }
    .ok_or_else(|| {
        ProcessError::Custom(format!(
            "Col2Im: runtime {name} must be a 1D int tensor of known length, got {:?}",
            arg.ty
        ))
    })?;
    Ok(Col2ImShape::Runtime {
        input: RuntimeInputRef::new(arg.name.clone(), index),
        len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    /// Helper to create a Col2Im test node
    fn create_test_node(
        image_shape: Vec<i64>,
        block_shape: Vec<i64>,
        input_static_shape: Option<Vec<usize>>,
        dilations: Option<Vec<i64>>,
        pads: Option<Vec<i64>>,
        strides: Option<Vec<i64>>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::Col2Im, "test_col2im")
            .input_tensor_f32("input", 3, input_static_shape)
            .input_tensor_i64_data("image_shape", image_shape.clone(), vec![image_shape.len()])
            .input_tensor_i64_data("block_shape", block_shape.clone(), vec![block_shape.len()])
            .output_tensor_f32("output", 0, None);

        if let Some(d) = dilations {
            builder = builder.attr_ints("dilations", d);
        }
        if let Some(p) = pads {
            builder = builder.attr_ints("pads", p);
        }
        if let Some(s) = strides {
            builder = builder.attr_ints("strides", s);
        }

        builder.build_with_graph_data(18)
    }

    #[test]
    fn test_runtime_shapes() {
        let shape_input = || ArgType::Tensor(TensorType::new_known(DType::I64, vec![2]));
        let mut node = TestNodeBuilder::new(NodeType::Col2Im, "test_col2im")
            .input_tensor_f32("input", 3, Some(vec![1, 4, 9]))
            .add_input("image_shape", shape_input())
            .add_input("block_shape", shape_input())
            .output_tensor_f32("output", 0, None)
            .build();

        let config = Col2ImProcessor.extract_config(&node, 18).unwrap();
        assert!(matches!(
            config.image_shape,
            Col2ImShape::Runtime { len: 2, ref input } if input.input_index == 1
        ));
        assert!(matches!(
            config.block_shape,
            Col2ImShape::Runtime { len: 2, .. }
        ));

        Col2ImProcessor
            .infer_types(&mut node, 18, &OutputPreferences::new())
            .unwrap();
        assert_eq!(
            node.outputs[0].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 4,
                static_shape: Some(vec![Some(1), None, None, None]),
            })
        );
    }

    #[test]
    fn test_basic_config_extraction() {
        let node = create_test_node(vec![5, 5], vec![2, 2], None, None, None, None);
        let processor = Col2ImProcessor;
        let config = processor.extract_config(&node, 18).unwrap();

        assert_eq!(config.image_shape, Col2ImShape::Static(vec![5, 5]));
        assert_eq!(config.block_shape, Col2ImShape::Static(vec![2, 2]));
        assert_eq!(config.dilations, vec![1, 1]);
        assert_eq!(config.pads, vec![0, 0, 0, 0]);
        assert_eq!(config.strides, vec![1, 1]);
    }

    #[test]
    fn test_config_with_custom_attributes() {
        let node = create_test_node(
            vec![5, 5],
            vec![2, 2],
            None,
            Some(vec![2, 2]),
            Some(vec![1, 1, 1, 1]),
            Some(vec![2, 2]),
        );
        let processor = Col2ImProcessor;
        let config = processor.extract_config(&node, 18).unwrap();

        assert_eq!(config.dilations, vec![2, 2]);
        assert_eq!(config.pads, vec![1, 1, 1, 1]);
        assert_eq!(config.strides, vec![2, 2]);
    }

    #[test]
    fn test_type_inference_basic() {
        // Input: [1, 20, 16] (batch=1, C*prod(block)=5*2*2=20, L=16)
        // image_shape: [5, 5], block_shape: [2, 2]
        // Output: [1, 5, 5, 5] (batch=1, C=20/4=5, H=5, W=5)
        let mut node = create_test_node(
            vec![5, 5],
            vec![2, 2],
            Some(vec![1, 20, 16]),
            None,
            None,
            None,
        );
        let processor = Col2ImProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 18, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4); // N + C + 2 spatial
                assert_eq!(
                    tensor.static_shape,
                    Some(vec![Some(1), Some(5), Some(5), Some(5)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_type_inference_dynamic_shape() {
        // Input without static shape
        let mut node = create_test_node(vec![5, 5], vec![2, 2], None, None, None, None);
        let processor = Col2ImProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 18, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
                assert_eq!(
                    tensor.static_shape,
                    Some(vec![None, None, Some(5), Some(5)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_type_inference_1d() {
        // 1D case: image_shape=[10], block_shape=[3]
        // Input: [1, 12, 8] (C*prod(block)=4*3=12)
        // Output: [1, 4, 10]
        let mut node = create_test_node(vec![10], vec![3], Some(vec![1, 12, 8]), None, None, None);
        let processor = Col2ImProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 18, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 3); // N + C + 1 spatial
                assert_eq!(tensor.static_shape, Some(vec![Some(1), Some(4), Some(10)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_invalid_input_rank() {
        // Create node with rank-2 input (should fail, needs rank 3)
        let builder = TestNodeBuilder::new(NodeType::Col2Im, "test_col2im")
            .input_tensor_f32("input", 2, None)
            .input_tensor_i64_data("image_shape", vec![5, 5], vec![2])
            .input_tensor_i64_data("block_shape", vec![2, 2], vec![2])
            .output_tensor_f32("output", 0, None);
        let mut node = builder.build_with_graph_data(18);

        let processor = Col2ImProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 18, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_spatial_dims() {
        // Test with 3D spatial dims (not supported yet)
        // Image [5, 5, 5], Block [2, 2, 2]
        let mut node = create_test_node(
            vec![5, 5, 5],
            vec![2, 2, 2],
            Some(vec![1, 20, 16]),
            None,
            None,
            None,
        );
        let processor = Col2ImProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 18, &prefs);
        assert!(result.is_err());
        match result {
            Err(ProcessError::Custom(msg)) => {
                assert!(
                    msg.contains("Col2Im currently only supports 1D and 2D spatial dimensions")
                );
            }
            _ => panic!("Expected Custom ProcessError, got {:?}", result),
        }
    }

    fn config_error(node: &RawNode) -> ProcessError {
        Col2ImProcessor.extract_config(node, 18).unwrap_err()
    }

    #[test]
    fn test_rejects_shape_length_mismatch() {
        let node = create_test_node(vec![5, 5], vec![2], None, None, None, None);
        assert!(
            matches!(config_error(&node), ProcessError::Custom(msg) if msg.contains("block_shape has 1"))
        );
    }

    #[test]
    fn test_rejects_non_positive_static_shape() {
        let node = create_test_node(vec![0, 5], vec![2, 2], None, None, None, None);
        assert!(
            matches!(config_error(&node), ProcessError::Custom(msg) if msg.contains("must be positive"))
        );
        let node = create_test_node(vec![5, 5], vec![2, -1], None, None, None, None);
        assert!(
            matches!(config_error(&node), ProcessError::Custom(msg) if msg.contains("must be positive"))
        );
    }

    #[test]
    fn test_rejects_attribute_length_mismatch() {
        let node = create_test_node(vec![5, 5], vec![2, 2], None, None, None, Some(vec![1]));
        assert!(matches!(
            config_error(&node),
            ProcessError::InvalidAttribute { ref name, .. } if name == "strides"
        ));
        let node = create_test_node(vec![5, 5], vec![2, 2], None, None, Some(vec![0, 0]), None);
        assert!(matches!(
            config_error(&node),
            ProcessError::InvalidAttribute { ref name, .. } if name == "pads"
        ));
    }

    #[test]
    fn test_rejects_out_of_range_attributes() {
        let node = create_test_node(
            vec![5, 5],
            vec![2, 2],
            None,
            None,
            Some(vec![0, -1, 0, 0]),
            None,
        );
        assert!(matches!(
            config_error(&node),
            ProcessError::InvalidAttribute { ref name, .. } if name == "pads"
        ));
        let node = create_test_node(vec![5, 5], vec![2, 2], None, Some(vec![1, 0]), None, None);
        assert!(matches!(
            config_error(&node),
            ProcessError::InvalidAttribute { ref name, .. } if name == "dilations"
        ));
    }
}
