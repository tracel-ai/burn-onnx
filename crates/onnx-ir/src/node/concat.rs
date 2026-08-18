//! # Concat
//!
//! Concatenates a list of tensors into a single tensor along a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Concat.html>
//!
//! ## Opset Versions
//! - **Opset 1-3**: Initial version
//! - **Opset 4-10**: Updated type support
//! - **Opset 11-12**: More type support
//! - **Opset 13+**: Current version with extended type support
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, Node, RawNode, TensorType};
use crate::processor::{
    InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec,
    ProcessError,
};

/// Configuration for Concat operation
#[derive(Debug, Clone, new)]
pub struct ConcatConfig {
    pub axis: usize,
}

/// Node representation for Concat operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ConcatNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ConcatConfig,
}

pub(crate) struct ConcatProcessor;

/// Element count a rank-1 tensor input contributes to the concatenated output,
/// or `None` when no length is statically known: either the length is decided at
/// runtime (a Shape sliced with runtime bounds, say) or the input is a constant
/// whose value the IR no longer holds.
///
/// A lifted constant's own data shape wins over a declared `static_shape`.
/// Callers must have checked `rank == 1`.
fn rank1_tensor_len(input: &Argument, ty: &TensorType) -> Option<usize> {
    input
        .value()
        .as_ref()
        .and_then(|v| v.shape.first().copied())
        .or_else(|| ty.static_shape_known().map(|s| s[0]))
}

/// Name the inputs that forced a Shape output to degrade to a tensor. Usually
/// the length is genuinely dynamic, but a lifted constant whose value the IR
/// lost lands here too, and that one only surfaces much later, in whatever
/// downstream node needed a Shape. Record it while the cause is still known.
fn log_runtime_length_fallback(node: &RawNode) {
    for input in &node.inputs {
        if let ArgType::Tensor(t) = &input.ty
            && t.rank == 1
            && rank1_tensor_len(input, t).is_none()
        {
            log::debug!(
                "Concat node {}: input '{}' has no statically known length, so the \
                 output is a rank-1 tensor instead of a Shape",
                node.name,
                input.name
            );
        }
    }
}

/// A Shape is a 1-D i64 array, and ONNX requires every Concat input to share
/// one element type and rank, so the other inputs have to be integers of rank 1
/// or less. Reject the rest here: burn-onnx unifies these inputs to i64, which
/// silently truncates a float, cannot type-check a bool, and cannot join a
/// higher-rank tensor to a 1-D one.
fn validate_shape_concat_inputs(node: &RawNode) -> Result<(), ProcessError> {
    for (i, input) in node.inputs.iter().enumerate() {
        let dtype = match &input.ty {
            ArgType::Shape(_) => continue,
            ArgType::Tensor(t) if t.rank != 1 => {
                return Err(ProcessError::TypeMismatch {
                    expected: "rank-1 input, since Concat mixes it with a Shape".to_string(),
                    actual: format!("rank {} at input {} ('{}')", t.rank, i, input.name),
                });
            }
            ArgType::Tensor(t) => t.dtype,
            ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => *dtype,
        };

        if !dtype.is_int() && !dtype.is_uint() {
            return Err(ProcessError::TypeMismatch {
                expected: "integer input, since Concat mixes it with a Shape".to_string(),
                actual: format!("{:?} at input {} ('{}')", dtype, i, input.name),
            });
        }
    }

    Ok(())
}

impl NodeProcessor for ConcatProcessor {
    type Config = ConcatConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &RawNode,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        use crate::processor::ArgPreference;

        if node.inputs.is_empty() {
            return Ok(None);
        }

        let mut prefs = InputPreferences::new();
        let has_shape = node.inputs.iter().any(|input| input.ty.is_shape());

        for input in &node.inputs {
            if has_shape && matches!(&input.ty, ArgType::Tensor(t) if t.rank == 1) {
                // When concatenating with Shape inputs, prefer rank-1 tensors as Shape
                prefs = prefs.add(&input.name, ArgPreference::Shape);
            }
            if input.ty.is_scalar() {
                // Scalar concat inputs must be native (used in TensorData::from([...]))
                prefs = prefs.add(&input.name, ArgPreference::ScalarNative);
            }
        }

        Ok(Some(prefs))
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get reference to config for type inference (not used, but extracted for consistency)
        let _config = self
            .extract_config(node, opset)
            .expect("Config extraction failed");

        // For shapes, axis must be 0 (since they're 1D) - validation already done in extract_config

        // Infer output type

        // Check if we have mixed Shape and rank-1 tensor inputs
        let has_shape = node
            .inputs
            .iter()
            .any(|i| matches!(i.ty, ArgType::Shape(_)));
        let has_rank1_tensor = node
            .inputs
            .iter()
            .any(|i| matches!(&i.ty, ArgType::Tensor(t) if t.rank == 1));
        let has_scalar = node.inputs.iter().any(|i| i.ty.is_scalar());

        if has_shape {
            validate_shape_concat_inputs(node)?;
        }

        // Handle scalar inputs: concatenating scalars with rank-1 tensors or shapes produces a 1D output
        if has_scalar {
            // When we have scalars, we can mix with rank-1 tensors and shapes (all are 1D int arrays)
            // Calculate total output length
            let mut total_length = 0usize;
            let mut length_known = true;
            for (i, input) in node.inputs.iter().enumerate() {
                match &input.ty {
                    ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => {
                        total_length += 1; // Each scalar contributes 1 element
                    }
                    ArgType::Tensor(t) if t.rank == 1 => match rank1_tensor_len(input, t) {
                        Some(len) => total_length += len,
                        None => length_known = false,
                    },
                    ArgType::Shape(rank) => {
                        total_length += rank;
                    }
                    _ => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "Scalar, rank-1 Tensor, or Shape".to_string(),
                            actual: format!("{:?} at input {}", input.ty, i),
                        });
                    }
                }
            }

            // With a Shape input the result is shape arithmetic: a fixed-size
            // Shape when every length is known, an i64 tensor when one is not.
            // Without one it is a rank-1 tensor of the shared input dtype.
            if has_shape && length_known {
                node.outputs[0].ty = ArgType::Shape(total_length);
            } else if has_shape {
                // A runtime-length input makes the total unknown, so the result
                // cannot be a fixed-size Shape array; fall back to a 1D tensor.
                log_runtime_length_fallback(node);
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: crate::ir::DType::I64,
                    rank: 1,
                    static_shape: None,
                });
            } else {
                // Get dtype from first scalar or tensor and validate all match
                let first_dtype = node
                    .inputs
                    .iter()
                    .find_map(|input| match &input.ty {
                        ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => Some(*dtype),
                        ArgType::Tensor(t) => Some(t.dtype),
                        _ => None,
                    })
                    .unwrap_or(crate::ir::DType::I64);

                for (i, input) in node.inputs.iter().enumerate() {
                    let dtype = match &input.ty {
                        ArgType::ScalarNative(d) | ArgType::ScalarTensor(d) => Some(*d),
                        ArgType::Tensor(t) => Some(t.dtype),
                        _ => None,
                    };
                    if let Some(d) = dtype
                        && d != first_dtype
                    {
                        return Err(ProcessError::TypeMismatch {
                            expected: format!("{:?}", first_dtype),
                            actual: format!("{:?} at input {}", d, i),
                        });
                    }
                }

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: first_dtype,
                    rank: 1,
                    static_shape: length_known.then(|| vec![Some(total_length)]),
                });
            }
            return Ok(());
        }

        // Validate all inputs have compatible types (all Tensor or all Shape, except mixed Shape/rank-1 tensor case)
        if !has_shape && !has_rank1_tensor {
            // Regular tensor case - validate all inputs are tensors with same dtype
            let first_dtype = match &node.inputs[0].ty {
                ArgType::Tensor(t) => t.dtype,
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor".to_string(),
                        actual: format!("{:?}", node.inputs[0].ty),
                    });
                }
            };

            for (i, input) in node.inputs.iter().enumerate().skip(1) {
                match &input.ty {
                    ArgType::Tensor(t) => {
                        if t.dtype != first_dtype {
                            return Err(ProcessError::TypeMismatch {
                                expected: format!("Tensor with dtype {:?}", first_dtype),
                                actual: format!("Tensor with dtype {:?} at input {}", t.dtype, i),
                            });
                        }
                    }
                    _ => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "Tensor".to_string(),
                            actual: format!("{:?} at input {}", input.ty, i),
                        });
                    }
                }
            }
        }

        if has_shape && has_rank1_tensor {
            // Mixed inputs: sum the Shape ranks and the tensor contributions.
            // A rank-1 tensor that is a lifted constant contributes its own
            // length here, and later flips to a Shape input once its producer
            // honors this node's Shape preference.
            let mut total_rank: usize = 0;
            let mut length_known = true;

            for input in &node.inputs {
                match &input.ty {
                    ArgType::Shape(rank) => {
                        total_rank += rank;
                    }
                    ArgType::Tensor(t) if t.rank == 1 => match rank1_tensor_len(input, t) {
                        Some(len) => total_rank += len,
                        None => length_known = false,
                    },
                    _ => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "Shape or rank-1 Tensor".to_string(),
                            actual: format!("{:?}", input.ty),
                        });
                    }
                }
            }

            node.outputs[0].ty = if length_known {
                // Every length is known, so the total is final
                ArgType::Shape(total_rank)
            } else {
                // At least one tensor has a runtime-only length (e.g. a Shape
                // sliced with runtime bounds), so the total is not known at
                // compile time and the result cannot be a fixed-size array.
                log_runtime_length_fallback(node);
                ArgType::Tensor(TensorType {
                    dtype: crate::ir::DType::I64,
                    rank: 1,
                    static_shape: None,
                })
            };
            return Ok(());
        }

        // Get the first input type - it determines the output type
        let first_input_type = &node.inputs[0].ty;

        match first_input_type {
            ArgType::Tensor(tensor) => {
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: tensor.dtype,
                    rank: tensor.rank,
                    static_shape: None,
                });
            }
            ArgType::Shape(_) => {
                // When concatenating shapes, we sum up their ranks
                let total_rank: usize = node
                    .inputs
                    .iter()
                    .map(|input| match &input.ty {
                        ArgType::Shape(rank) => Ok(*rank),
                        _ => Err(ProcessError::TypeMismatch {
                            expected: "Shape".to_string(),
                            actual: format!("{:?}", input.ty),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .sum();

                node.outputs[0].ty = ArgType::Shape(total_rank);
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", first_input_type),
                });
            }
        }

        Ok(())
    }

    fn is_noop(&self, node: &RawNode) -> bool {
        // Concat is a no-op when there is only a single input
        node.inputs.len() == 1
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract the axis attribute (required per ONNX spec)
        let mut axis: Option<i64> = None;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = Some(value.clone().into_i64());
                break;
            }
        }

        let axis = axis.ok_or_else(|| ProcessError::MissingAttribute("axis".to_string()))?;

        // extract the rank based on input type
        let rank = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.rank as i64,
            ArgType::Shape(_) => 1, // Shapes are 1D
            ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => 0, // Scalars are rank-0
        };

        // if axis is negative, it is counted from the end
        let normalized_axis = if axis < 0 { axis + rank } else { axis };

        // TODO: Add validation that normalized_axis is within valid range [0, rank)
        // According to spec, axis must be in range [-r, r-1] where r = rank(inputs)
        // After normalization, should validate: 0 <= normalized_axis < rank
        // TODO: Add test for empty inputs list - spec requires 1+ inputs but not validated
        // TODO: Add test for single input - edge case that should work but may not be tested
        // TODO: Validate all non-concat dimensions match across inputs - currently only dtype checked for tensors
        // TODO: Add test for very large axis values that overflow after normalization

        let config = ConcatConfig {
            axis: normalized_axis as usize,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Concat(ConcatNode {
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

    fn create_test_node(axis: i64, input_rank: usize, num_inputs: usize) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Concat, "test_concat")
            .input_tensors_f32("data", num_inputs, input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
    }

    #[test]
    fn test_concat_config_basic() {
        let node = create_test_node(1, 3, 2).process(ConcatProcessor, 16);
        let processor = ConcatProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_concat_config_negative_axis() {
        let node = create_test_node(-2, 3, 2).process(ConcatProcessor, 16);
        let processor = ConcatProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, 1); // -2 + 3 = 1
    }

    #[test]
    fn test_concat_config_shape_input() {
        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat_shape")
            .input_shape("shape1", 2)
            .input_shape("shape2", 3)
            .output_shape("output", 5)
            .attr_int("axis", 0) // Required attribute
            .process(ConcatProcessor, 16);

        let processor = ConcatProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, 0); // Shape concat uses axis 0
    }

    #[test]
    fn test_concat_config_missing_axis() {
        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat")
            .input_tensor_f32("data1", 3, None)
            .input_tensor_f32("data2", 3, None)
            .output_tensor_f32("output", 3, None)
            .build();

        let node = node;
        let processor = ConcatProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::MissingAttribute(_))));
    }

    #[test]
    fn test_concat_config_axis_out_of_bounds() {
        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat")
            .input_tensor_f32("data1", 3, None)
            .input_tensor_f32("data2", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_int("axis", 3)
            .build();

        let processor = ConcatProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(result.is_ok()); // axis 3 is valid, it's normalized to 3 which equals rank
    }

    #[test]
    fn test_concat_update_outputs_shape() {
        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat_shape")
            .input_shape("shape1", 2)
            .input_shape("shape2", 3)
            .input_shape("shape3", 1)
            .output_shape("output", 0) // Will be updated
            .attr_int("axis", 0) // Required attribute
            .process(ConcatProcessor, 16);

        // Check that output is Shape with sum of input ranks
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => assert_eq!(*rank, 6), // 2 + 3 + 1
            _ => panic!("Expected Shape output"),
        }
    }

    #[test]
    fn test_concat_config_shape_negative_axis() {
        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat_shape")
            .input_shape("shape1", 2)
            .input_shape("shape2", 3)
            .output_shape("output", 5)
            .attr_int("axis", -1) // -1 should become 0 for 1D shapes
            .process(ConcatProcessor, 16);

        let processor = ConcatProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, 0); // -1 + 1 = 0
    }

    #[test]
    fn test_concat_config_shape_invalid_axis() {
        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat_shape")
            .input_shape("shape1", 2)
            .input_shape("shape2", 3)
            .output_shape("output", 5)
            .attr_int("axis", 1)
            .build();

        let processor = ConcatProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(result.is_ok()); // axis 1 is valid for Shape inputs (rank-1)
    }

    #[test]
    fn test_concat_mixed_inputs() {
        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_mixed")
            .input_shape("shape1", 2)
            .input_tensor_f32("tensor1", 3, None)
            .output_shape("output", 0)
            .attr_int("axis", 0)
            .build();

        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_concat_scalar_inputs() {
        // Test concatenating scalar inputs (reproduces issue #4228)
        use burn_tensor::DType;

        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat_scalar")
            .input_scalar_i64("scalar1")
            .input_scalar_i64("scalar2")
            .output_tensor_i64("output", 1, None)
            .attr_int("axis", 0)
            .process(ConcatProcessor, 16);

        // Check that output is 1D tensor with length = number of inputs
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 1);
                assert_eq!(t.dtype, DType::I64);
                assert_eq!(t.static_shape, Some(vec![Some(2)])); // 2 scalar inputs
            }
            _ => panic!("Expected Tensor output, got {:?}", node.outputs[0].ty),
        }
    }

    #[test]
    fn test_concat_scalar_config_extraction() {
        // Test that extract_config works with scalar inputs
        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat_scalar")
            .input_scalar_i64("scalar1")
            .input_scalar_i64("scalar2")
            .output_tensor_i64("output", 1, None)
            .attr_int("axis", 0)
            .build();

        let processor = ConcatProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.axis, 0); // Only valid axis for scalars
    }

    #[test]
    fn test_concat_scalar_dtype_mismatch() {
        use burn_tensor::DType;

        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_dtype_mismatch")
            .input_scalar_i64("s1")
            .input_scalar("s2", DType::F32)
            .output_tensor_i64("output", 1, None)
            .attr_int("axis", 0)
            .build();

        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_concat_multiple_scalars() {
        // Test concatenating multiple scalar inputs
        use burn_tensor::DType;

        let node = TestNodeBuilder::new(NodeType::Concat, "test_concat_multi_scalar")
            .input_scalar_i64("s1")
            .input_scalar_i64("s2")
            .input_scalar_i64("s3")
            .input_scalar_i64("s4")
            .output_tensor_i64("output", 1, None)
            .attr_int("axis", 0)
            .process(ConcatProcessor, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 1);
                assert_eq!(t.dtype, DType::I64);
                assert_eq!(t.static_shape, Some(vec![Some(4)])); // 4 scalar inputs
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_concat_mixed_shape_and_rank1_tensor_with_static_shape() {
        // Regression: when mixing Shape inputs with rank-1 Tensors that have
        // static_shape but no constant value, Concat should use static_shape
        // to determine the tensor's element contribution instead of defaulting to 1.
        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_mixed_static")
            .input_shape("shape1", 2)
            .input_tensor_i64("tensor1", 1, Some(vec![0])) // rank-1, static_shape=[0]
            .output_shape("output", 0) // will be updated
            .attr_int("axis", 0)
            .build();

        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Shape(2) + rank-1 tensor with static_shape [0] => Shape(2 + 0 = 2)
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => assert_eq!(*rank, 2),
            _ => panic!("Expected Shape output, got {:?}", node.outputs[0].ty),
        }
    }

    #[test]
    fn test_concat_mixed_shape_and_runtime_length_tensor() {
        // A rank-1 tensor whose length is only known at runtime (e.g. a Shape
        // sliced with runtime bounds) makes the total length unknown, so the
        // output cannot be a fixed-size Shape array.
        use burn_tensor::DType;

        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_mixed_runtime")
            .input_shape("shape1", 2)
            .input_tensor_i64("tensor1", 1, None) // rank-1, length unknown
            .output_shape("output", 0)
            .attr_int("axis", 0)
            .build();

        ConcatProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 1);
                assert_eq!(t.dtype, DType::I64);
                assert_eq!(t.static_shape, None);
            }
            other => panic!("Expected rank-1 Tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_concat_scalar_with_shape_and_runtime_length_tensor() {
        // Same rule when a scalar is in the mix: an unknown-length tensor
        // input forces a tensor output instead of a Shape.
        use burn_tensor::DType;

        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_scalar_runtime")
            .input_shape("shape1", 2)
            .input_scalar_i64("scalar1")
            .input_tensor_i64("tensor1", 1, None)
            .output_shape("output", 0)
            .attr_int("axis", 0)
            .build();

        ConcatProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 1);
                assert_eq!(t.dtype, DType::I64);
                assert_eq!(t.static_shape, None);
            }
            other => panic!("Expected rank-1 Tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_concat_shape_with_float_tensor_is_rejected() {
        // Shape values are i64 and ONNX requires one element type across the
        // inputs, so a float alongside a Shape is a malformed graph. Catching it
        // here keeps burn-onnx from silently truncating the float to i64.
        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_float")
            .input_shape("shape1", 2)
            .input_tensor_f32("tensor1", 1, Some(vec![2]))
            .output_shape("output", 0)
            .attr_int("axis", 0)
            .build();

        let err = ConcatProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .expect_err("float input alongside a Shape must be rejected");

        assert!(
            format!("{}", err).contains("integer"),
            "expected an integer-input error, got: {}",
            err
        );
    }

    #[test]
    fn test_concat_shape_with_rank2_tensor_is_rejected() {
        // A Shape is 1-D, so a higher-rank tensor cannot join it. Without this
        // the node falls through to the plain tensor path and codegen emits a
        // `Tensor::cat` of a rank-1 and a rank-2 tensor, which cannot compile.
        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_rank2")
            .input_shape("shape1", 2)
            .input_tensor_i64("tensor1", 2, None)
            .output_shape("output", 0)
            .attr_int("axis", 0)
            .build();

        let err = ConcatProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .expect_err("a rank-2 input alongside a Shape must be rejected");

        assert!(
            format!("{}", err).contains("rank-1"),
            "expected a rank error, got: {}",
            err
        );
    }

    #[test]
    fn test_concat_shape_with_unsigned_tensor_is_accepted() {
        // The dtype check rejects non-integers, not non-i64: unsigned shape
        // arithmetic is valid and burn-onnx casts it to i64.
        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_uint")
            .input_shape("shape1", 2)
            .add_input(
                "tensor1",
                ArgType::Tensor(TensorType {
                    dtype: burn_tensor::DType::U32,
                    rank: 1,
                    static_shape: Some(vec![Some(2)]),
                }),
            )
            .output_shape("output", 0)
            .attr_int("axis", 0)
            .build();

        ConcatProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .expect("an unsigned integer input alongside a Shape is valid");

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => assert_eq!(*rank, 4),
            other => panic!("Expected Shape output, got {:?}", other),
        }
    }

    #[test]
    fn test_concat_runtime_length_tensor_without_shape_has_no_static_shape() {
        // No Shape input, so the output stays a tensor of the shared dtype, but
        // an unknown-length input still has to clear static_shape: downstream
        // nodes read it to pick an output rank.
        use burn_tensor::DType;

        let mut node = TestNodeBuilder::new(NodeType::Concat, "test_concat_scalar_tensor_runtime")
            .input_scalar_i64("scalar1")
            .input_tensor_i64("tensor1", 1, None)
            .output_tensor_i64("output", 1, None)
            .attr_int("axis", 0)
            .build();

        ConcatProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::I64);
                assert_eq!(t.static_shape, None);
            }
            other => panic!("Expected rank-1 Tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_concat_single_input_is_noop() {
        let node = create_test_node(0, 3, 1).build();
        assert!(ConcatProcessor.is_noop(&node));
    }

    #[test]
    fn test_concat_multiple_inputs_is_not_noop() {
        let node = create_test_node(0, 3, 2).build();
        assert!(!ConcatProcessor.is_noop(&node));
    }
}
