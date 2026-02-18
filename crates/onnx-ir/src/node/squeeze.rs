//! # Squeeze
//!
//! Removes single-dimensional entries from the shape of a tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Squeeze.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with optional 'axes' attribute.
//! - **Opset 11**: Clarified semantics and behavior for negative axis values.
//! - **Opset 13**: Changed 'axes' from attribute to optional input, enabling dynamic axes specification at runtime.
//!
//! This implementation supports all opset versions. For opset < 13, axes are read from the
//! `axes` attribute. For opset 13+, axes are read from the second input.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::ir::{ArgType, Argument, Node, RawNode, RuntimeInputRef, TensorDataExt, TensorType};

/// Represents either a static value or a runtime argument for squeeze axes.
#[derive(Debug, Clone)]
pub enum SqueezeInput {
    /// Static axes known at compile time.
    Static(Vec<i64>),
    /// Runtime axes determined during execution.
    Runtime(RuntimeInputRef),
}

impl Default for SqueezeInput {
    fn default() -> Self {
        SqueezeInput::Static(vec![])
    }
}

/// Configuration for Squeeze operation
#[derive(Debug, Clone, new)]
pub struct SqueezeConfig {
    pub axes: Option<SqueezeInput>,
}

/// Node representation for Squeeze operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SqueezeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SqueezeConfig,
}

pub(crate) struct SqueezeProcessor;

impl NodeProcessor for SqueezeProcessor {
    type Config = SqueezeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, opset: usize) -> Result<(), ProcessError> {
        // Lift axes input (input[1]) if present (opset 13+ only; opset < 13 uses attribute)
        if opset >= 13
            && node.inputs.len() > 1
            && !node.inputs[1].is_optional()
            && node.inputs[1].is_constant()
        {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get reference to config for type inference
        let config = self
            .extract_config(node, opset)
            .expect("Config extraction failed");
        let axes = config.axes.clone();

        // Extract axes for type inference
        let axes_vec = match &axes {
            Some(SqueezeInput::Static(axes_vec)) => Some(axes_vec.clone()),
            Some(SqueezeInput::Runtime(_)) => None,
            None => None,
        };

        // TODO: Missing validation that axes values are in valid range [-rank, rank-1].
        // Out-of-bounds axes should be rejected but aren't validated here.

        // TODO: Missing validation that axes doesn't contain duplicates.
        // Duplicate axes should be rejected per ONNX spec but not validated.

        match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => {
                let output_rank = match axes_vec {
                    None => {
                        // When axes is None, ONNX spec squeezes all dimensions of size 1
                        if let Some(ref static_shape) = tensor.static_shape {
                            static_shape.iter().filter(|dim| **dim != Some(1)).count()
                        } else {
                            return Err(ProcessError::Custom(
                                "Squeeze: Cannot infer output rank when axes is None and input tensor static shape is unknown".to_string()
                            ));
                        }
                    }
                    Some(ref axes_vec) => {
                        // Validate that we're not trying to squeeze more axes than the tensor has
                        if axes_vec.len() > tensor.rank {
                            return Err(ProcessError::Custom(format!(
                                "Squeeze: Cannot squeeze {} axes from a rank {} tensor",
                                axes_vec.len(),
                                tensor.rank
                            )));
                        }

                        // TODO: Missing validation that squeezed dimensions actually have size 1.
                        // ONNX spec requires dimensions to be size 1 to be squeezed, but implementation
                        // doesn't validate this when static_shape is available. Should check:
                        // for &axis in axes_vec { assert static_shape[axis] == 1 }

                        tensor.rank - axes_vec.len()
                    }
                };

                // Compute output static_shape by removing squeezed dimensions
                let static_shape = {
                    let input_shape = tensor
                        .static_shape
                        .clone()
                        .unwrap_or_else(|| vec![None; tensor.rank]);
                    match axes_vec {
                        None => {
                            // Squeeze all dims of size 1
                            Some(
                                input_shape
                                    .iter()
                                    .filter(|dim| **dim != Some(1))
                                    .copied()
                                    .collect(),
                            )
                        }
                        Some(ref axes_vec) => {
                            // Normalize axes and remove those positions
                            let rank = tensor.rank as i64;
                            let remove: Vec<usize> = axes_vec
                                .iter()
                                .map(|&a| {
                                    if a < 0 {
                                        (a + rank) as usize
                                    } else {
                                        a as usize
                                    }
                                })
                                .collect();
                            Some(
                                input_shape
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, _)| !remove.contains(i))
                                    .map(|(_, dim)| *dim)
                                    .collect(),
                            )
                        }
                    }
                };

                // When all dimensions are squeezed (rank=0), keep as ScalarTensor (on device)
                // Downstream consumers that need native will request ScalarNative via preferences
                node.outputs[0].ty = if output_rank == 0 {
                    ArgType::ScalarTensor(tensor.dtype)
                } else {
                    ArgType::Tensor(TensorType {
                        dtype: tensor.dtype,
                        rank: output_rank,
                        static_shape,
                    })
                };
            }
            ArgType::Shape(shape_rank) => {
                if let Some(ref axes_vec) = axes_vec
                    && !axes_vec.is_empty()
                    && (axes_vec.len() != 1 || axes_vec[0] != 0)
                {
                    return Err(ProcessError::Custom(format!(
                        "Squeeze on Shape input only supports squeezing axis 0, got axes: {:?}",
                        axes_vec
                    )));
                }

                if *shape_rank == 1 {
                    node.outputs[0].ty = ArgType::ScalarNative(crate::ir::DType::I64);
                } else {
                    node.outputs[0].ty = ArgType::Shape(*shape_rank);
                }
            }
            ArgType::ScalarTensor(scalar_type) | ArgType::ScalarNative(scalar_type) => {
                node.outputs[0].ty = ArgType::ScalarNative(*scalar_type);
            }
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, opset: usize) -> Result<Self::Config, ProcessError> {
        // Check axes attribute (valid in opset < 13)
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axes" {
                if opset >= 13 {
                    return Err(ProcessError::Custom(
                        "Squeeze: axes must be provided as input (not attribute) in opset 13+"
                            .to_string(),
                    ));
                }
                let axes = value.clone().into_i64s();
                if axes.is_empty() {
                    return Ok(SqueezeConfig { axes: None });
                }
                return Ok(SqueezeConfig {
                    axes: Some(SqueezeInput::Static(axes)),
                });
            }
        }

        // Fall through to input-based extraction (opset 13+ or opset < 13 without attribute)
        fn get_squeeze_axes(node: &RawNode) -> Option<SqueezeInput> {
            if node.inputs.len() < 2 {
                return None; // No axes means squeeze all dims with size 1
            }

            let input = &node.inputs[1];
            match input.value() {
                None => {
                    // Runtime input - no static value available
                    Some(SqueezeInput::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        1,
                    )))
                }
                Some(value) => match value.to_i64_vec() {
                    Ok(axes) => Some(SqueezeInput::Static(axes)),
                    Err(_) => None, // Invalid type
                },
            }
        }

        let axes = get_squeeze_axes(node);
        let config = SqueezeConfig { axes };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Squeeze(SqueezeNode {
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

    fn create_test_node(axes: Option<Vec<i64>>, rank: usize) -> TestNodeBuilder {
        let output_rank = if let Some(ref axes_vec) = axes {
            rank - axes_vec.len()
        } else {
            // When no axes specified, we don't know how many dims will be squeezed
            // without static shape info, but for testing we'll assume same as input
            rank
        };

        let mut builder = TestNodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("squeezed", output_rank, None);

        // Add axes as a second input (ONNX opset 13+ style)
        if let Some(axes_val) = axes {
            builder = builder.input_tensor_i64_data("axes", axes_val.clone(), vec![axes_val.len()]);
        }

        builder
    }

    fn create_runtime_squeeze_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Squeeze, "test_runtime_squeeze")
            .input_tensor_f32("data", 4, Some(vec![2, 3, 4, 5])) // Need some shape
            .input_tensor_i64("axes", 0, None) // Runtime input - no static value
            .output_tensor_f32("squeezed", 2, None)
    }

    #[test]
    fn test_squeeze_config_with_axes_input() {
        let node = create_test_node(Some(vec![0, 2]), 4).build_with_graph_data(16);
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(config.axes, Some(SqueezeInput::Static(ref axes)) if axes == &vec![0, 2]));
    }

    #[test]
    fn test_squeeze_config_no_axes_input() {
        // Test with no axes input - need static shape with dims of size 1
        let node = TestNodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", 4, Some(vec![2, 1, 3, 1])) // Has two dims of size 1
            .output_tensor_f32("squeezed", 2, None) // Will squeeze to rank 2
            .build();
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(config.axes.is_none());
    }

    #[test]
    fn test_squeeze_config_runtime_axes() {
        let node = create_runtime_squeeze_node().build();
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(config.axes, Some(SqueezeInput::Runtime(ref arg)) if arg.name == "axes"));
    }

    // TODO: Missing test for squeezing dimension that is not size 1 - should fail.
    // E.g., input shape [2, 1, 3], axes=[0] should fail because dim 0 has size 2, not 1.

    // TODO: Missing test for negative axes normalization and validation.
    // E.g., axes=[-1] for rank-3 should squeeze last dimension.

    // TODO: Missing test for duplicate axes - axes=[0, 0] should be rejected.

    // TODO: Missing test for out-of-bounds axes - axes=[5] for rank-3 should be rejected.

    // TODO: Missing test for opset < 13 behavior - axes as attribute vs input.
    // Implementation requires opset 13+ but this transition isn't tested.

    #[test]
    fn test_squeeze_all_dims_to_scalar() {
        // Test squeezing all dimensions produces Scalar, not Tensor(rank=0)
        // This maintains consistency with proto conversion
        let node = create_test_node(Some(vec![0]), 1).build_with_graph_data(16);
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Verify output is ScalarTensor (stays on device)
        match &node.outputs[0].ty {
            ArgType::ScalarTensor(dtype) => {
                assert_eq!(*dtype, crate::ir::DType::F32);
            }
            other => panic!("Expected ScalarTensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_squeeze_propagates_static_shape() {
        // Input [2, 1, 3, 1] with axes [1, 3] -> output [2, 3]
        let mut node = TestNodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", 4, Some(vec![2, 1, 3, 1]))
            .input_tensor_i64_data("axes", vec![1, 3], vec![2])
            .output_tensor_f32("squeezed", 2, None)
            .build_with_graph_data(16);

        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(2), Some(3)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_squeeze_no_axes_removes_size_1() {
        // Input [2, 1, 3, 1] with no axes -> squeeze all 1s -> [2, 3]
        let mut node = TestNodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", 4, Some(vec![2, 1, 3, 1]))
            .output_tensor_f32("squeezed", 2, None)
            .build();

        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(2), Some(3)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_squeeze_no_static_shape_with_axes() {
        // Input rank 4, no static_shape, axes [1] -> output rank 3, all dims unknown
        let node = create_test_node(Some(vec![1]), 4).build_with_graph_data(16);
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                // Even without input static_shape, produces partial shape
                assert_eq!(t.static_shape, Some(vec![None, None, None]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_squeeze_negative_axes() {
        // Input [2, 1, 3] with axes [-2] -> output [2, 3]
        let mut node = TestNodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", 3, Some(vec![2, 1, 3]))
            .input_tensor_i64_data("axes", vec![-2], vec![1])
            .output_tensor_f32("squeezed", 2, None)
            .build_with_graph_data(16);

        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(2), Some(3)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
