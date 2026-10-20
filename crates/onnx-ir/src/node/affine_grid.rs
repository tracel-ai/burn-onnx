//! # AffineGrid
//!
//! Generates a 2D or 3D sampling grid from a batch of affine matrices `theta`.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__AffineGrid.html>
//!
//! ## Opset Versions
//! - **Opset 20**: Initial version with theta, size inputs and the align_corners attribute.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, RuntimeInputRef, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, validate_opset,
};

const OP_NAME: &str = "AffineGrid";

/// The target size `(N, C, H, W)` (2D) or `(N, C, D, H, W)` (3D) of the grid.
#[derive(Debug, Clone)]
pub enum AffineGridSize {
    /// Size known at compile time, as the ONNX values.
    Static(Vec<i64>),
    /// Size determined at runtime from a graph input.
    Runtime(RuntimeInputRef),
}

impl Default for AffineGridSize {
    fn default() -> Self {
        Self::Static(vec![1, 1, 1, 1])
    }
}

/// Configuration for the AffineGrid operation.
#[derive(Debug, Clone, Default)]
pub struct AffineGridConfig {
    /// If true, -1 and 1 refer to the centers of the corner pixels; otherwise to their
    /// outer edges.
    pub align_corners: bool,
    /// The target size, with 4 values (2D) or 5 values (3D).
    pub size: AffineGridSize,
}

/// Node representation for AffineGrid operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct AffineGridNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: AffineGridConfig,
}

pub(crate) struct AffineGridProcessor;

/// The static values of the `size` input, when it is a constant.
fn static_size(node: &RawNode) -> Result<Option<Vec<i64>>, ProcessError> {
    let Some(data) = node.inputs[1].value() else {
        return Ok(None);
    };
    data.try_into_vec()
        .map(Some)
        .map_err(|e| ProcessError::Custom(format!("{OP_NAME}: cannot read size: {e:?}")))
}

/// The output batch and spatial sizes of a constant `size`, checked like ONNX Runtime:
/// N must be non-negative and every spatial size positive; C is not used.
fn checked_size(size: &[i64]) -> Result<(usize, Vec<usize>), ProcessError> {
    let batch = usize::try_from(size[0]).map_err(|_| {
        ProcessError::Custom(format!(
            "{OP_NAME}: N must be non-negative, got size {size:?}"
        ))
    })?;
    let spatial = size[2..]
        .iter()
        .map(|&dim| usize::try_from(dim).ok().filter(|&dim| dim > 0))
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| {
            ProcessError::Custom(format!(
                "{OP_NAME}: spatial sizes must be positive, got size {size:?}"
            ))
        })?;
    Ok((batch, spatial))
}

/// Number of values in `size` (4 for 2D, 5 for 3D), from its value, its type or the
/// shape of `theta` (`(N, 2, 3)` or `(N, 3, 4)`).
fn size_len(node: &RawNode) -> Result<usize, ProcessError> {
    if let Some(size) = static_size(node)? {
        return Ok(size.len());
    }
    let from_size = match &node.inputs[1].ty {
        ArgType::Shape(len) => Some(*len),
        ArgType::Tensor(tensor) if tensor.rank == 1 => tensor
            .static_shape
            .as_ref()
            .and_then(|shape| shape.first().copied().flatten()),
        _ => None,
    };
    let from_theta = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor
            .static_shape
            .as_ref()
            .and_then(|shape| shape.get(1).copied().flatten())
            .map(|spatial| spatial + 2),
        _ => None,
    };
    from_size.or(from_theta).ok_or_else(|| {
        ProcessError::Custom(format!(
            "{OP_NAME}: cannot tell 2D from 3D: the length of size and the shape of theta are unknown"
        ))
    })
}

impl NodeProcessor for AffineGridProcessor {
    type Config = AffineGridConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 20,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        if node.inputs[1].is_constant() {
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
        validate_opset(opset, 20)?;

        let theta = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) if tensor.rank == 3 => tensor.clone(),
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "theta as a rank-3 tensor (N, 2, 3) or (N, 3, 4)".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };
        if !theta.dtype.is_float() {
            return Err(ProcessError::TypeMismatch {
                expected: "float theta".to_string(),
                actual: format!("{:?}", theta.dtype),
            });
        }
        match &node.inputs[1].ty {
            ArgType::Shape(_) => {}
            ArgType::Tensor(tensor) if tensor.rank == 1 => {}
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "size as a rank-1 tensor".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        }
        // A Shape input holds i64 values and has no dtype of its own
        if !matches!(&node.inputs[1].ty, ArgType::Shape(_)) {
            let size_dtype = node.inputs[1].ty.elem_type();
            if size_dtype != DType::I64 {
                return Err(ProcessError::TypeMismatch {
                    expected: "int64 size".to_string(),
                    actual: format!("{size_dtype:?}"),
                });
            }
        }

        let len = size_len(node)?;
        if len != 4 && len != 5 {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: size must have 4 (2D) or 5 (3D) values, got {len}"
            )));
        }
        let spatial = len - 2;
        // theta is (N, 2, 3) in 2D and (N, 3, 4) in 3D
        if let Some(shape) = &theta.static_shape {
            let expected = [Some(spatial), Some(spatial + 1)];
            for (axis, (&actual, expected)) in shape[1..].iter().zip(expected).enumerate() {
                if actual.is_some() && actual != expected {
                    return Err(ProcessError::Custom(format!(
                        "{OP_NAME}: theta must have shape (N, {spatial}, {}) for size with {len} values, got {:?} on axis {}",
                        spatial + 1,
                        actual,
                        axis + 1
                    )));
                }
            }
        }

        // Output: (N, H, W, 2) or (N, D, H, W, 3)
        let batch = theta
            .static_shape
            .as_ref()
            .and_then(|shape| shape.first().copied().flatten());
        let static_shape = match static_size(node)? {
            Some(size) => {
                let (size_batch, spatial_sizes) = checked_size(&size)?;
                if let Some(batch) = batch
                    && batch != size_batch
                {
                    return Err(ProcessError::Custom(format!(
                        "{OP_NAME}: theta has batch {batch} but size has N = {size_batch}"
                    )));
                }
                let mut shape: Vec<Option<usize>> = vec![Some(size_batch)];
                shape.extend(spatial_sizes.into_iter().map(Some));
                shape.push(Some(spatial));
                shape
            }
            None => {
                let mut shape = vec![batch];
                shape.extend(vec![None; spatial]);
                shape.push(Some(spatial));
                shape
            }
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: theta.dtype,
            rank: len,
            static_shape: Some(static_shape),
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let align_corners = node
            .attrs
            .get("align_corners")
            .map(|value| value.clone().into_i64().map(|x| x != 0))
            .transpose()?
            .unwrap_or(false);

        let size = match static_size(node)? {
            Some(size) => AffineGridSize::Static(size),
            None => AffineGridSize::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1)),
        };

        Ok(AffineGridConfig {
            align_corners,
            size,
        })
    }

    fn build_node(&self, mut builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        // Drop the size input if static (baked into config).
        if matches!(config.size, AffineGridSize::Static(_)) {
            builder.inputs.truncate(1);
        }

        Ok(Node::AffineGrid(AffineGridNode {
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
    use crate::processor::OutputPreferences;

    fn output_type(node: &RawNode) -> TensorType {
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            other => panic!("Expected a tensor output, got {other:?}"),
        }
    }

    /// AffineGrid with a theta of the given shape and a runtime int64 size of
    /// `size_len` values (unknown length when `None`).
    fn runtime_node(theta: Option<Vec<usize>>, size_len: Option<usize>) -> RawNode {
        TestNodeBuilder::new(NodeType::AffineGrid, "test_affine_grid")
            .input_tensor_f32("theta", 3, theta)
            .input_tensor_i64("size", 1, size_len.map(|len| vec![len]))
            .output_tensor_f32("grid", 0, None)
            .build()
    }

    /// AffineGrid with a theta of the given shape and a constant size.
    fn static_node(theta: Vec<usize>, size: Vec<i64>) -> RawNode {
        let len = size.len();
        TestNodeBuilder::new(NodeType::AffineGrid, "test_affine_grid")
            .input_tensor_f32("theta", 3, Some(theta))
            .input_tensor_i64_data("size", size, vec![len])
            .output_tensor_f32("grid", 0, None)
            .attr_int("align_corners", 1)
            .build_with_graph_data(20)
    }

    fn infer(node: &mut RawNode) -> Result<(), ProcessError> {
        AffineGridProcessor.infer_types(node, 20, &OutputPreferences::new())
    }

    #[test]
    fn runtime_size_2d() {
        let mut node = runtime_node(Some(vec![2, 2, 3]), Some(4));
        infer(&mut node).unwrap();
        let output = output_type(&node);
        assert_eq!(output.rank, 4);
        assert_eq!(output.dtype, DType::F32);
        // N comes from theta, H and W from the runtime size, 2 coordinates per point
        assert_eq!(
            output.static_shape,
            Some(vec![Some(2), None, None, Some(2)])
        );

        let config = AffineGridProcessor.extract_config(&node, 20).unwrap();
        assert!(!config.align_corners);
        assert!(
            matches!(&config.size, AffineGridSize::Runtime(r) if r.name == "size" && r.input_index == 1)
        );
    }

    #[test]
    fn static_size_3d() {
        let mut node = static_node(vec![1, 3, 4], vec![1, 1, 2, 3, 2]);
        infer(&mut node).unwrap();
        let output = output_type(&node);
        assert_eq!(output.rank, 5);
        assert_eq!(
            output.static_shape,
            Some(vec![Some(1), Some(2), Some(3), Some(2), Some(3)])
        );

        let config = AffineGridProcessor.extract_config(&node, 20).unwrap();
        assert!(config.align_corners);
        assert!(matches!(&config.size, AffineGridSize::Static(size) if size == &[1, 1, 2, 3, 2]));

        // The constant size is baked into the config, so the node keeps only theta
        let built = AffineGridProcessor.build_node(node, 20).unwrap();
        match built {
            Node::AffineGrid(node) => assert_eq!(node.inputs.len(), 1),
            other => panic!("Expected an AffineGrid node, got {other:?}"),
        }
    }

    #[test]
    fn dimension_from_theta_when_size_length_is_unknown() {
        // theta (N, 3, 4) is 3D, so size has 5 values and the grid rank 5
        let mut node = runtime_node(Some(vec![1, 3, 4]), None);
        infer(&mut node).unwrap();
        assert_eq!(output_type(&node).rank, 5);
    }

    #[test]
    fn dimension_from_a_shape_size() {
        let mut node = TestNodeBuilder::new(NodeType::AffineGrid, "test_affine_grid")
            .input_tensor_f32("theta", 3, None)
            .input_shape("size", 4)
            .output_tensor_f32("grid", 0, None)
            .build();
        infer(&mut node).unwrap();
        let output = output_type(&node);
        assert_eq!(output.rank, 4);
        assert_eq!(output.static_shape, Some(vec![None, None, None, Some(2)]));
    }

    #[test]
    fn unknown_dimension_is_rejected() {
        let mut node = runtime_node(None, None);
        let err = infer(&mut node).unwrap_err();
        assert!(
            matches!(&err, ProcessError::Custom(msg) if msg.contains("cannot tell 2D from 3D")),
            "{err}"
        );
    }

    #[test]
    fn size_length_must_be_4_or_5() {
        let mut node = runtime_node(None, Some(3));
        let err = infer(&mut node).unwrap_err();
        assert!(
            matches!(&err, ProcessError::Custom(msg) if msg.contains("4 (2D) or 5 (3D) values, got 3")),
            "{err}"
        );
    }

    #[test]
    fn theta_shape_must_match_size_length() {
        // size has 4 values (2D) but theta is (N, 3, 4) (3D)
        let mut node = runtime_node(Some(vec![1, 3, 4]), Some(4));
        let err = infer(&mut node).unwrap_err();
        assert!(
            matches!(&err, ProcessError::Custom(msg) if msg.contains("theta must have shape (N, 2, 3)")),
            "{err}"
        );
    }

    #[test]
    fn theta_batch_must_match_constant_size() {
        let mut node = static_node(vec![2, 2, 3], vec![3, 1, 4, 4]);
        let err = infer(&mut node).unwrap_err();
        assert!(
            matches!(&err, ProcessError::Custom(msg) if msg.contains("theta has batch 2 but size has N = 3")),
            "{err}"
        );
    }

    #[test]
    fn constant_size_is_checked_like_onnx_runtime() {
        // A zero or negative spatial size is rejected
        for size in [vec![1, 1, 0, 4], vec![1, 1, -2, 4]] {
            let mut node = static_node(vec![1, 2, 3], size);
            let err = infer(&mut node).unwrap_err();
            assert!(
                matches!(&err, ProcessError::Custom(msg) if msg.contains("spatial sizes must be positive")),
                "{err}"
            );
        }

        // A negative N is rejected
        let mut node = static_node(vec![1, 2, 3], vec![-1, 1, 2, 4]);
        let err = infer(&mut node).unwrap_err();
        assert!(
            matches!(&err, ProcessError::Custom(msg) if msg.contains("N must be non-negative")),
            "{err}"
        );

        // C is not used, so any value is accepted, as in ONNX Runtime
        let mut node = static_node(vec![1, 2, 3], vec![1, -1, 2, 4]);
        infer(&mut node).unwrap();
        assert_eq!(
            output_type(&node).static_shape,
            Some(vec![Some(1), Some(2), Some(4), Some(2)])
        );
    }

    #[test]
    fn theta_must_be_a_rank_3_float_tensor() {
        let mut node = TestNodeBuilder::new(NodeType::AffineGrid, "test_affine_grid")
            .input_tensor_f32("theta", 2, None)
            .input_tensor_i64("size", 1, Some(vec![4]))
            .output_tensor_f32("grid", 0, None)
            .build();
        assert!(matches!(
            infer(&mut node),
            Err(ProcessError::TypeMismatch { .. })
        ));

        let mut node = TestNodeBuilder::new(NodeType::AffineGrid, "test_affine_grid")
            .input_tensor_i64("theta", 3, None)
            .input_tensor_i64("size", 1, Some(vec![4]))
            .output_tensor_f32("grid", 0, None)
            .build();
        let err = infer(&mut node).unwrap_err();
        assert!(
            matches!(&err, ProcessError::TypeMismatch { expected, .. } if expected == "float theta"),
            "{err}"
        );
    }

    #[test]
    fn size_must_be_a_rank_1_tensor() {
        let mut node = TestNodeBuilder::new(NodeType::AffineGrid, "test_affine_grid")
            .input_tensor_f32("theta", 3, Some(vec![1, 2, 3]))
            .input_tensor_i64("size", 2, Some(vec![2, 2]))
            .output_tensor_f32("grid", 0, None)
            .build();
        let err = infer(&mut node).unwrap_err();
        assert!(
            matches!(&err, ProcessError::TypeMismatch { expected, .. } if expected == "size as a rank-1 tensor"),
            "{err}"
        );
    }

    #[test]
    fn size_must_be_int64() {
        let mut node = TestNodeBuilder::new(NodeType::AffineGrid, "test_affine_grid")
            .input_tensor_f32("theta", 3, None)
            .input_tensor_i32("size", 1, Some(vec![4]))
            .output_tensor_f32("grid", 0, None)
            .build();
        let err = infer(&mut node).unwrap_err();
        assert!(
            matches!(&err, ProcessError::TypeMismatch { expected, .. } if expected == "int64 size"),
            "{err}"
        );
    }

    #[test]
    fn opset_below_20_is_rejected() {
        let mut node = runtime_node(Some(vec![1, 2, 3]), Some(4));
        assert!(
            AffineGridProcessor
                .infer_types(&mut node, 19, &OutputPreferences::new())
                .is_err()
        );
    }
}
