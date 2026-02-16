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

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
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

/// Configuration for Col2Im operation
#[derive(Debug, Clone, new)]
pub struct Col2ImConfig {
    /// Image shape (spatial dimensions of the output image)
    pub image_shape: Vec<usize>,
    /// Block shape (kernel size)
    pub block_shape: Vec<usize>,
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
        let static_shape = if let Some(input_shape) = &tensor.static_shape {
            // Full inference if input shape is fully known
            let n = input_shape[0];
            let block_product: usize = config.block_shape.iter().product();
            let c = input_shape[1].map(|v| v / block_product);

            let mut shape = vec![n, c];
            for &dim in &config.image_shape {
                shape.push(Some(dim));
            }
            Some(shape)
        } else {
            // Partial inference: N, C unknown, but spatial dims known from config
            let mut shape = vec![None, None]; // N, C
            for &dim in &config.image_shape {
                shape.push(Some(dim));
            }
            Some(shape)
        };

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
        use crate::ir::TensorDataExt;

        // Extract image_shape from input[1] (required, must be constant)
        let image_shape = match node.inputs[1].value() {
            Some(data) => data
                .to_i64_vec()
                .map_err(|_| {
                    ProcessError::Custom("Col2Im: image_shape must be int64 tensor".to_string())
                })?
                .iter()
                .map(|&v| v as usize)
                .collect::<Vec<_>>(),
            None => {
                return Err(ProcessError::Custom(
                    "Col2Im: image_shape (input[1]) must be a constant".to_string(),
                ));
            }
        };

        // Extract block_shape from input[2] (required, must be constant)
        let block_shape = match node.inputs[2].value() {
            Some(data) => data
                .to_i64_vec()
                .map_err(|_| {
                    ProcessError::Custom("Col2Im: block_shape must be int64 tensor".to_string())
                })?
                .iter()
                .map(|&v| v as usize)
                .collect::<Vec<_>>(),
            None => {
                return Err(ProcessError::Custom(
                    "Col2Im: block_shape (input[2]) must be a constant".to_string(),
                ));
            }
        };

        let num_spatial_dims = image_shape.len();

        // Note: ONNX spec requires num_spatial_dims >= 2, but we support 1D as an extension.

        // Extract dilations attribute (default: all 1s)
        let dilations = node
            .attrs
            .get("dilations")
            .map(|v| {
                v.clone()
                    .into_i64s()
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![1; num_spatial_dims]);

        // Extract pads attribute (default: all 0s, format is [begin, end] per dim)
        let pads = node
            .attrs
            .get("pads")
            .map(|v| {
                v.clone()
                    .into_i64s()
                    .iter()
                    .map(|&p| p as usize)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![0; num_spatial_dims * 2]);

        // Extract strides attribute (default: all 1s)
        let strides = node
            .attrs
            .get("strides")
            .map(|v| {
                v.clone()
                    .into_i64s()
                    .iter()
                    .map(|&s| s as usize)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![1; num_spatial_dims]);

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
    fn test_basic_config_extraction() {
        let node = create_test_node(vec![5, 5], vec![2, 2], None, None, None, None);
        let processor = Col2ImProcessor;
        let config = processor.extract_config(&node, 18).unwrap();

        assert_eq!(config.image_shape, vec![5, 5]);
        assert_eq!(config.block_shape, vec![2, 2]);
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
}
