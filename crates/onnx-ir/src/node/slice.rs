//! # Slice
//!
//! Extracts a slice from the input tensor along multiple axes.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Slice.html>
//!
//! ## Type Constraints
//!
//! - T: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
//! - Tind: tensor(int32), tensor(int64)
//!
//! ## Opset Versions
//!
//! - **Opset 1-9**: `starts`, `ends`, and `axes` were attributes (static only).
//! - **Opset 10**: **BREAKING CHANGE** - Changed `starts`, `ends`, and `axes` from attributes to inputs
//!   for dynamic slicing support. This enables runtime determination of slice parameters.
//! - **Opset 11**: Added optional `steps` input for strided slicing.
//! - **Opset 13**: Added bfloat16 and additional type support.
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, RuntimeInputRef, TensorDataExt};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for the Slice operation.
#[derive(Debug, Clone, new)]
pub struct SliceConfig {
    pub starts: SliceInput,
    pub ends: SliceInput,
    pub axes: Option<SliceInput>,
    pub steps: Option<SliceInput>,
}

/// Node representation for Slice operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SliceNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SliceConfig,
}

/// Represents either a static value or a runtime argument for slice parameters.
#[derive(Debug, Clone)]
pub enum SliceInput {
    /// Static value known at compile time.
    Static(Vec<i64>),
    /// Runtime argument determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl Default for SliceInput {
    fn default() -> Self {
        Self::Static(Vec::new())
    }
}

/// Normalize negative axes to positive indices based on tensor rank.
fn normalize_axes(axes: &mut [i64], rank: usize, _node_name: &str) {
    for axis in axes.iter_mut() {
        if *axis < 0 {
            *axis += rank as i64;
        }
    }
}

/// Calculate output length for slicing across a dimension.
///
/// Follows ONNX/Python semantics: negative indices count from the end, and
/// out-of-range indices clamp. The ONNX-specific sentinels `i64::MAX` (end with
/// step > 0, "to end") and `i64::MIN` (end with step < 0, "past first element")
/// are handled implicitly by clamping — for `step > 0`, `i64::MAX` clamps to
/// `dim_len`; for `step < 0`, `i64::MIN + dim_len` clamps to `-1`.
fn calculate_dim_slice_output_len(start: i64, end: i64, step: i64, dim_size: usize) -> usize {
    if dim_size == 0 {
        return 0;
    }
    let dim_len = dim_size as i64;

    // Normalize an index: negative values count from the end (idx + dim_len),
    // then the result is clamped into the valid range for this step direction.
    fn normalize_index(idx: i64, lo: i64, hi: i64, n: i64) -> i64 {
        let abs = if idx < 0 { idx + n } else { idx };
        abs.clamp(lo, hi)
    }

    // For step > 0, valid endpoints are [0, dim_len]: start at dim_len means
    // empty slice; end at dim_len means slice to the last element.
    // For step < 0, start must reference a real index [0, dim_len-1] (we cannot
    // begin iterating from past the end), and end ranges over [-1, dim_len-1]
    // where -1 represents the conceptual position before the first element.
    let (norm_start, norm_end) = if step > 0 {
        (
            normalize_index(start, 0, dim_len, dim_len),
            normalize_index(end, 0, dim_len, dim_len),
        )
    } else {
        (
            normalize_index(start, 0, dim_len - 1, dim_len),
            normalize_index(end, -1, dim_len - 1, dim_len),
        )
    };

    let range_len = (norm_end - norm_start) * step.signum();
    if range_len <= 0 {
        0
    } else {
        ((range_len + step.abs() - 1) / step.abs()) as usize
    }
}

pub(crate) struct SliceProcessor;

impl NodeProcessor for SliceProcessor {
    type Config = SliceConfig;

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
    ) -> Result<Option<crate::processor::InputPreferences>, ProcessError> {
        use crate::processor::{ArgPreference, InputPreferences};

        let mut prefs = InputPreferences::new();
        for input in node.inputs.iter().skip(1) {
            // Prefer constants to be Shape; scalars must be native (for Slice bounds)
            prefs = prefs.add(&input.name, ArgPreference::Shape);
            if input.ty.is_scalar() {
                prefs = prefs.add(&input.name, ArgPreference::ScalarNative);
            }
        }

        Ok(Some(prefs))
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift starts input (input[1]) if present
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        // Lift ends input (input[2]) if present
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        // Lift axes input (input[3]) if present
        if node.inputs.len() > 3 && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }

        // Lift steps input (input[4]) if present
        if node.inputs.len() > 4 && node.inputs[4].is_constant() {
            node.inputs[4].to_static()?;
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
        let config = self.extract_config(node, opset)?;

        // Infer output type based on input type
        let input_ty = node.inputs[0].ty.clone();

        match input_ty {
            ArgType::Tensor(tensor_type) => {
                // Slice changes dimension sizes along sliced axes. Initialize all dimensions
                // with sizes from input, then overwrite the axes we can compute statically.
                let mut static_shape = tensor_type
                    .static_shape
                    .clone()
                    .unwrap_or_else(|| vec![None; tensor_type.rank]);

                if let (
                    SliceInput::Static(starts),
                    SliceInput::Static(ends),
                    Some(SliceInput::Static(axes)),
                    Some(SliceInput::Static(steps)),
                ) = (&config.starts, &config.ends, &config.axes, &config.steps)
                {
                    for (i, &axis) in axes.iter().enumerate() {
                        let axis = axis as usize;
                        if let Some(Some(dim_size)) =
                            tensor_type.static_shape.as_ref().and_then(|s| s.get(axis))
                        {
                            let out_dim = calculate_dim_slice_output_len(
                                starts[i], ends[i], steps[i], *dim_size,
                            );
                            static_shape[axis] = Some(out_dim);
                        }
                    }
                }

                node.outputs[0].ty = ArgType::Tensor(crate::ir::TensorType {
                    dtype: tensor_type.dtype,
                    rank: tensor_type.rank,
                    static_shape: Some(static_shape),
                });
            }
            ArgType::Shape(shape_rank) => {
                // Runtime path: output_len cannot be computed without the bound
                // values, so degrade to a rank-1 i64 tensor.
                let static_bounds = match (&config.starts, &config.ends, &config.steps) {
                    (SliceInput::Static(s), SliceInput::Static(e), steps_opt) => {
                        let steps = match steps_opt {
                            Some(SliceInput::Static(st)) => st.clone(),
                            Some(SliceInput::Runtime(_)) => vec![],
                            None => vec![1],
                        };
                        if steps.is_empty() {
                            None
                        } else {
                            Some((s, e, steps))
                        }
                    }
                    _ => None,
                };

                if let Some((starts, ends, steps)) = static_bounds {
                    if starts.len() != 1 || ends.len() != 1 {
                        return Err(ProcessError::Custom(format!(
                            "Slice on Shape input requires exactly one dimension slice config for node {}",
                            node.name
                        )));
                    }

                    let step = if steps.is_empty() { 1 } else { steps[0] };
                    let output_len =
                        calculate_dim_slice_output_len(starts[0], ends[0], step, shape_rank);
                    node.outputs[0].ty = ArgType::Shape(output_len);
                } else {
                    // Enforce single-axis, step=1 invariants up front so the
                    // codegen can rely on them.
                    let len_of = |input: &SliceInput| -> Option<usize> {
                        match input {
                            SliceInput::Static(v) => Some(v.len()),
                            SliceInput::Runtime(r) => match &node.inputs[r.input_index].ty {
                                ArgType::Tensor(t) => t
                                    .static_shape
                                    .as_ref()
                                    .and_then(|s| s.first().copied().flatten()),
                                ArgType::Shape(n) => Some(*n),
                                ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => Some(1),
                            },
                        }
                    };
                    let mut any_unknown_len = false;
                    for (which, input) in [("starts", &config.starts), ("ends", &config.ends)] {
                        match len_of(input) {
                            Some(n) if n != 1 => {
                                return Err(ProcessError::Custom(format!(
                                    "Slice on Shape input requires single-axis slicing; node {} has {} of length {}",
                                    node.name, which, n
                                )));
                            }
                            None => any_unknown_len = true,
                            _ => {}
                        }
                    }
                    if any_unknown_len {
                        log::debug!(
                            "Slice node {}: runtime bound length not statically known; \
                             codegen will assert exactly one element at inference time",
                            node.name
                        );
                    }
                    match &config.steps {
                        Some(SliceInput::Static(steps)) if steps.iter().any(|&s| s != 1) => {
                            return Err(ProcessError::Custom(format!(
                                "Slice on Shape input with runtime bounds only supports step=1; node {} has steps={:?}",
                                node.name, steps
                            )));
                        }
                        Some(SliceInput::Runtime(_)) => {
                            return Err(ProcessError::Custom(format!(
                                "Slice on Shape input with runtime steps is not supported (node {})",
                                node.name
                            )));
                        }
                        _ => {}
                    }
                    node.outputs[0].ty = ArgType::Tensor(crate::ir::TensorType {
                        dtype: crate::ir::DType::I64,
                        rank: 1,
                        static_shape: None,
                    });
                }
            }
            ArgType::ScalarTensor(dtype) => {
                // Treat as rank-1 tensor; slicing may change element count
                node.outputs[0].ty = ArgType::Tensor(crate::ir::TensorType {
                    dtype,
                    rank: 1,
                    static_shape: None,
                });
            }
            unsupported_type => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor, Shape, or ScalarTensor".to_string(),
                    actual: format!("{:?}", unsupported_type),
                });
            }
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // For opset < 10, starts/ends/axes are attributes
        if node.inputs.len() < 2 {
            let starts = node
                .attrs
                .get("starts")
                .map(|v| SliceInput::Static(v.clone().into_i64s()))
                .ok_or_else(|| ProcessError::MissingAttribute("starts".to_string()))?;
            let ends = node
                .attrs
                .get("ends")
                .map(|v| SliceInput::Static(v.clone().into_i64s()))
                .ok_or_else(|| ProcessError::MissingAttribute("ends".to_string()))?;
            let mut axes = node
                .attrs
                .get("axes")
                .map(|v| SliceInput::Static(v.clone().into_i64s()));

            // Apply default axes if not provided
            if axes.is_none()
                && let SliceInput::Static(ref starts_vec) = starts
            {
                axes = Some(SliceInput::Static((0..starts_vec.len() as i64).collect()));
            }

            // Normalize negative axes
            if let Some(SliceInput::Static(ref mut axes_values)) = axes
                && let ArgType::Tensor(ref tensor_type) = node.inputs[0].ty
            {
                normalize_axes(axes_values, tensor_type.rank, &node.name);
            }

            // No steps in opset < 10
            let steps = if let SliceInput::Static(ref starts_vec) = starts {
                Some(SliceInput::Static(vec![1; starts_vec.len()]))
            } else {
                None
            };

            return Ok(SliceConfig {
                starts,
                ends,
                axes,
                steps,
            });
        }

        // Extract config - helper function to get slice inputs (opset 10+)
        fn get_slice_input(
            node: &RawNode,
            index: usize,
        ) -> Result<Option<SliceInput>, ProcessError> {
            let input = match node.inputs.get(index) {
                Some(i) => i,
                None => return Ok(None),
            };

            // Check if this is a Shape type (preferred for Slice parameters)
            if matches!(input.ty, ArgType::Shape(_)) {
                // Try to get static value if available
                match input.value() {
                    Some(tensor_data) => match tensor_data.to_i64_vec() {
                        Ok(vec) => return Ok(Some(SliceInput::Static(vec))),
                        Err(_) => {
                            return Err(ProcessError::Custom(format!(
                                "Slice Shape input at index {} must be int32 or int64",
                                index
                            )));
                        }
                    },
                    None => {
                        // Shape type without value means it's a runtime Shape (e.g., from Shape node)
                        // Runtime input - store reference instead of cloning the argument
                        return Ok(Some(SliceInput::Runtime(RuntimeInputRef::new(
                            input.name.clone(),
                            index,
                        ))));
                    }
                }
            }

            // Otherwise, handle as Tensor (backward compatibility)
            match input.value() {
                None => {
                    // Runtime input - store reference instead of cloning the argument
                    Ok(Some(SliceInput::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        index,
                    ))))
                }
                Some(tensor_data) => match tensor_data.to_i64_vec() {
                    Ok(vec) => Ok(Some(SliceInput::Static(vec))),
                    Err(_) => Err(ProcessError::Custom(format!(
                        "Slice input at index {} must be int32 or int64",
                        index
                    ))),
                },
            }
        }

        let starts = get_slice_input(node, 1)?
            .ok_or_else(|| ProcessError::MissingInput("starts".to_string()))?;

        let ends = get_slice_input(node, 2)?
            .ok_or_else(|| ProcessError::MissingInput("ends".to_string()))?;

        let axes = get_slice_input(node, 3)?;
        let steps = get_slice_input(node, 4)?;

        // Apply ONNX spec defaults for optional parameters.
        //
        // The number of slices is `len(starts)`. We determine it at IR time
        // either from a static starts vector (opset < 10 attributes, or a
        // constant tensor input) or from the starts argument's static_shape
        // (a 1-D tensor whose first dim is known). If neither is available
        // and `axes` is absent, we cannot fabricate a correct default, so
        // the codegen must not guess — returning None here forces the
        // downstream consumer to either supply axes or refuse the model.
        let num_slices: Option<usize> = match &starts {
            SliceInput::Static(v) => Some(v.len()),
            SliceInput::Runtime(r) => match &node.inputs[r.input_index].ty {
                ArgType::Tensor(t) => t
                    .static_shape
                    .as_ref()
                    .and_then(|s| s.first().copied().flatten()),
                ArgType::Shape(n) => Some(*n),
                _ => None,
            },
        };

        let (mut axes, steps) = if let Some(n) = num_slices.filter(|n| *n > 0) {
            // If steps not provided, default to all 1s (ONNX spec)
            let steps = if steps.is_none() {
                Some(SliceInput::Static(vec![1; n]))
            } else {
                steps
            };

            // If axes not provided, default to [0, 1, ..., n-1] (ONNX spec)
            let axes = if axes.is_none() {
                Some(SliceInput::Static((0..n as i64).collect()))
            } else {
                axes
            };

            (axes, steps)
        } else {
            // Length unknown: propagate as-is. If axes is None here, the
            // codegen will reject at emission rather than silently slicing
            // the wrong number of dimensions.
            (axes, steps)
        };

        // Validate steps if present - zeros are not allowed
        if let Some(SliceInput::Static(ref step_values)) = steps
            && step_values.contains(&0)
        {
            return Err(ProcessError::Custom(
                "Slice: step values cannot be zero".to_string(),
            ));
        }

        // TODO: Missing validation that starts, ends, axes, and steps have same length.
        // ONNX spec requires: len(starts) == len(ends) == len(axes) == len(steps).
        // Implementation doesn't validate this constraint, which could cause indexing errors.

        // TODO: Missing validation that axes values are unique and in valid range [-rank, rank-1].
        // Duplicate axes or out-of-range axes should be rejected but aren't validated.

        // Normalize negative axes if we have static axes and know the input rank
        if let Some(SliceInput::Static(ref mut axes_values)) = axes
            && let ArgType::Tensor(ref tensor_type) = node.inputs[0].ty
        {
            normalize_axes(axes_values, tensor_type.rank, &node.name);
        }

        let config = SliceConfig {
            starts,
            ends,
            axes,
            steps,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Slice(SliceNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    use super::*;

    fn create_test_node(
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Option<Vec<i64>>,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Slice, "test_slice")
            .input_tensor_f32("data", 3, None)
            .output_default("output");

        // Add inputs as tensors
        builder = builder.input_tensor_i64_data("starts", starts.clone(), vec![starts.len()]);
        builder = builder.input_tensor_i64_data("ends", ends.clone(), vec![ends.len()]);

        if let Some(axes_vec) = axes.clone() {
            builder = builder.input_tensor_i64_data("axes", axes_vec.clone(), vec![axes_vec.len()]);
        }

        builder
    }

    fn create_static_shape_slice_node(
        static_shape: Vec<Option<usize>>,
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Vec<i64>,
        steps: Vec<i64>,
    ) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_tensor_slice")
            .add_input(
                "data",
                ArgType::Tensor(crate::ir::TensorType {
                    dtype: crate::ir::DType::F32,
                    rank: static_shape.len(),
                    static_shape: Some(static_shape),
                }),
            )
            .input_tensor_i64_data("starts", starts.clone(), vec![starts.len()])
            .input_tensor_i64_data("ends", ends.clone(), vec![ends.len()])
            .input_tensor_i64_data("axes", axes.clone(), vec![axes.len()])
            .input_tensor_i64_data("steps", steps.clone(), vec![steps.len()])
            .output_default("output")
    }

    fn create_shape_input_node(start: i64, end: i64) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_slice_shape")
            .input_shape("data", 5)
            .input_tensor_i64_data("starts", vec![start], vec![1])
            .input_tensor_i64_data("ends", vec![end], vec![1])
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .output_default("output")
    }

    fn create_runtime_slice_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_runtime_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("starts", 0, None) // No static value - runtime input
            .input_tensor_i64("ends", 0, None) // No static value - runtime input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
    }

    fn create_mixed_slice_node_runtime_start() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_mixed_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("starts", 0, None) // Runtime input
            .input_tensor_i64_data("ends", vec![3], vec![1]) // Static input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
    }

    fn create_mixed_slice_node_runtime_end() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_mixed_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![1], vec![1]) // Static input
            .input_tensor_i64("ends", 0, None) // Runtime input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
    }

    #[test]
    fn test_slice_config_basic() {
        // Create a node with inputs for basic slicing
        let node =
            create_test_node(vec![1, 0], vec![3, 2], Some(vec![0, 2])).build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = processor.extract_config(&node, 16).unwrap();

        // Check that we have static starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Static(starts), SliceInput::Static(ends)) => {
                assert_eq!(starts, &vec![1, 0]);
                assert_eq!(ends, &vec![3, 2]);
                // Check axes
                if let Some(SliceInput::Static(axes)) = &result.axes {
                    assert_eq!(axes, &vec![0, 2]);
                }
                // Steps should have ONNX spec default of [1, 1] when not provided
                if let Some(SliceInput::Static(steps)) = &result.steps {
                    assert_eq!(steps, &vec![1, 1]);
                } else {
                    panic!("Expected steps to have ONNX spec default");
                }
            }
            _ => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_negative_axes() {
        // Test with negative axes values
        let node = create_test_node(vec![1], vec![3], Some(vec![-3])).build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = processor.extract_config(&node, 16).unwrap();

        // Check that we have static starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Static(starts), SliceInput::Static(ends)) => {
                assert_eq!(starts, &vec![1]);
                assert_eq!(ends, &vec![3]);
                // Check axes (should be normalized from -3 to 0 for rank 3 tensor)
                if let Some(SliceInput::Static(axes)) = &result.axes {
                    assert_eq!(axes, &vec![0]); // -3 + 3 = 0
                }
            }
            _ => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_default_axes() {
        // Test the default axes behavior (when axes input is not provided)
        let node = create_test_node(vec![1, 2], vec![3, 4], None).build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = processor.extract_config(&node, 16).unwrap();

        // Check that we have static starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Static(starts), SliceInput::Static(ends)) => {
                assert_eq!(starts, &vec![1, 2]);
                assert_eq!(ends, &vec![3, 4]);
                // axes should have ONNX spec default of [0, 1] when not provided
                if let Some(SliceInput::Static(axes)) = &result.axes {
                    assert_eq!(axes, &vec![0, 1]);
                } else {
                    panic!("Expected axes to have ONNX spec default");
                }
            }
            _ => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_runtime() {
        // Test with runtime inputs (no static values)
        let node = create_runtime_slice_node().build();

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = processor.extract_config(&node, 16).unwrap();

        // Check that we have runtime starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Runtime(starts), SliceInput::Runtime(ends)) => {
                assert_eq!(starts.name, "starts");
                assert_eq!(ends.name, "ends");
                // Check axes and steps
                if let Some(SliceInput::Static(axes)) = &result.axes {
                    assert_eq!(axes, &vec![0]);
                }
                if let Some(SliceInput::Static(steps)) = &result.steps {
                    assert_eq!(steps, &vec![1]);
                }
            }
            _ => panic!("Expected runtime config"),
        }
    }

    #[test]
    fn test_slice_update_output_rank_tensor_input() {
        // Test when input is a Tensor - output should preserve the same type
        let mut node = create_test_node(vec![1, 2], vec![3, 4], None).build_with_graph_data(16);

        // Before calling, input is Tensor and output is default
        assert!(matches!(node.inputs[0].ty, ArgType::Tensor(_)));
        assert!(matches!(node.outputs[0].ty, ArgType::Tensor(_)));

        let processor = SliceProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // After calling, output should be the same type as input
        assert!(
            matches!(&node.outputs[0].ty, ArgType::Tensor(tensor_type) if tensor_type.dtype == DType::F32 && tensor_type.rank == 3)
        );
    }

    #[test]
    fn test_slice_update_output_rank_shape_input() {
        // Test when input is a Shape - output should be a rank-1 Int64 Tensor
        let mut node = create_shape_input_node(1, 3).build_with_graph_data(16);

        // Before calling, input is Shape and output is default
        assert!(matches!(node.inputs[0].ty, ArgType::Shape(5)));
        // Default output type is Tensor with rank 0
        assert!(matches!(node.outputs[0].ty, ArgType::Tensor(ref t) if t.rank == 0));

        let processor = SliceProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // After calling, output should be ArgType::Shape with the calculated length
        // start = 1, end = 3 => output_len = 3 - 1 = 2
        assert!(matches!(&node.outputs[0].ty, ArgType::Shape(2)));
    }

    #[test]
    fn test_slice_shape_input_runtime_bounds_degrades_to_tensor() {
        // Slicing a Shape with runtime starts/ends. Output length cannot be
        // computed at IR time, so the type degrades to a rank-1 i64 tensor.
        let mut node = TestNodeBuilder::new(NodeType::Slice, "test_slice_shape_runtime")
            .input_shape("data", 5)
            .input_tensor_i64("starts", 1, None)
            .input_tensor_i64("ends", 1, None)
            .output_default("output")
            .build_with_graph_data(16);

        let processor = SliceProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::I64);
                assert_eq!(t.rank, 1);
                assert!(t.static_shape.is_none());
            }
            other => panic!("Expected Tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_slice_shape_input_runtime_bounds_length_gt_1_rejected() {
        // Bounds with a statically known length > 1 cannot work for Shape
        // slicing (codegen reads element 0 only). Reject early with a clear
        // error rather than silently producing the wrong slice.
        let mut node = TestNodeBuilder::new(NodeType::Slice, "shape_runtime_len2")
            .input_shape("data", 5)
            .input_tensor_i64("starts", 1, Some(vec![2]))
            .input_tensor_i64("ends", 1, Some(vec![2]))
            .output_default("output")
            .build_with_graph_data(16);

        let processor = SliceProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(ref m)) if m.contains("single-axis")));
    }

    #[test]
    fn test_slice_shape_input_runtime_bounds_step_2_rejected() {
        // The runtime Shape-slice codegen uses an implicit step of 1.
        // A non-1 step would produce a wrong slice silently, so reject it.
        let mut node = TestNodeBuilder::new(NodeType::Slice, "shape_runtime_step2")
            .input_shape("data", 5)
            .input_tensor_i64("starts", 1, None)
            .input_tensor_i64("ends", 1, None)
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![2], vec![1])
            .output_default("output")
            .build_with_graph_data(16);

        let processor = SliceProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(ref m)) if m.contains("step=1")));
    }

    #[test]
    fn test_slice_config_mixed_runtime_start() {
        // Test with runtime start but static end
        let node = create_mixed_slice_node_runtime_start().build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = processor.extract_config(&node, 16).unwrap();

        // Check that we have mixed starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Runtime(starts), SliceInput::Static(ends)) => {
                assert_eq!(starts.name, "starts");
                assert_eq!(ends, &vec![3]);
            }
            _ => panic!("Expected mixed config with runtime start and static end"),
        }
    }

    #[test]
    fn test_slice_config_mixed_runtime_end() {
        // Test with static start but runtime end
        let node = create_mixed_slice_node_runtime_end().build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = processor.extract_config(&node, 16).unwrap();

        // Check that we have mixed starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Static(starts), SliceInput::Runtime(ends)) => {
                assert_eq!(starts, &vec![1]);
                assert_eq!(ends.name, "ends");
            }
            _ => panic!("Expected mixed config with static start and runtime end"),
        }
    }

    #[test]
    fn test_slice_config_with_steps() {
        // Create a node with steps input
        let builder = TestNodeBuilder::new(NodeType::Slice, "test_slice_with_steps")
            .input_tensor_f32("data", 3, None)
            .input_tensor_i64_data("starts", vec![0, 0], vec![2])
            .input_tensor_i64_data("ends", vec![10, 10], vec![2])
            .input_tensor_i64_data("axes", vec![0, 1], vec![2])
            .input_tensor_i64_data("steps", vec![2, 3], vec![2])
            .output_default("output");

        let node = builder.build_with_graph_data(16);
        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = processor.extract_config(&node, 16).unwrap();

        // Check that we have static starts, ends, and steps
        match (&result.starts, &result.ends, &result.steps) {
            (
                SliceInput::Static(starts),
                SliceInput::Static(ends),
                Some(SliceInput::Static(steps)),
            ) => {
                assert_eq!(starts, &vec![0, 0]);
                assert_eq!(ends, &vec![10, 10]);
                assert_eq!(steps, &vec![2, 3]);
            }
            _ => panic!("Expected static config with steps"),
        }
    }

    #[test]
    fn test_slice_config_zero_step() {
        // Create a node with zero step value (should return error)
        let builder = TestNodeBuilder::new(NodeType::Slice, "test_zero_step")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![0], vec![1])
            .input_tensor_i64_data("ends", vec![10], vec![1])
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![0], vec![1])
            .output_default("output");

        let node = builder.build_with_graph_data(16);
        let node = node;

        let processor = SliceProcessor;

        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_slice_config_negative_steps() {
        // Create a node with negative step values
        let builder = TestNodeBuilder::new(NodeType::Slice, "test_negative_steps")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![0, 2], vec![2])
            .input_tensor_i64_data("ends", vec![10, 8], vec![2])
            .input_tensor_i64_data("axes", vec![0, 1], vec![2])
            .input_tensor_i64_data("steps", vec![-1, -2], vec![2])
            .output_default("output");

        let node = builder.build_with_graph_data(16);
        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = processor.extract_config(&node, 16).unwrap();

        // Check that negative steps are preserved
        match &result.steps {
            Some(SliceInput::Static(steps)) => {
                assert_eq!(steps, &vec![-1, -2]);
            }
            _ => panic!("Expected static steps with negative values"),
        }
    }

    #[test]
    fn test_tensor_static_shape_known_dim_filled() {
        // Slice axis 0: start=2, end=7, step=1
        let mut node = create_static_shape_slice_node(
            vec![Some(10), Some(20)],
            vec![2],
            vec![7],
            vec![0],
            vec![1],
        )
        .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor_type) => {
                let shape = tensor_type.static_shape.as_ref().unwrap();
                assert_eq!(shape[0], Some(5));
                assert_eq!(shape[1], Some(20));
            }
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_static_shape_multi_axis_partial_known() {
        // Slice axis 0: size=10, start=1, end=6, step=1
        // Slice axis 2: size=12, start=0, end=12, step=2
        let mut node = create_static_shape_slice_node(
            vec![Some(10), Some(5), Some(12)],
            vec![1, 0],
            vec![6, 12],
            vec![0, 2],
            vec![1, 2],
        )
        .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor_type) => {
                let shape = tensor_type.static_shape.as_ref().unwrap();
                assert_eq!(shape[0], Some(5));
                assert_eq!(shape[1], Some(5));
                assert_eq!(shape[2], Some(6));
            }
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_static_shape_runtime_inputs_all_none() {
        // Slice axis 0: size=None, start=None, end=None, step=1
        let mut node = create_runtime_slice_node().build();

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor_type) => match &tensor_type.static_shape {
                Some(shape) => {
                    assert_eq!(shape.len(), 2);
                    assert!(shape.iter().all(|d| d.is_none()));
                }
                None => panic!("Expect tensor static shape"),
            },
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_static_shape_end_i64_max_slices_to_end() {
        // Slice axis 0: size=8, start=3, end=i64::MAX, step=1
        let mut node = create_static_shape_slice_node(
            vec![Some(8)],
            vec![3],
            vec![i64::MAX],
            vec![0],
            vec![1],
        )
        .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                let shape = t.static_shape.as_ref().unwrap();
                assert_eq!(shape[0], Some(5));
            }
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_static_shape_end_i64_min_slices_to_start() {
        // Slice axis 0: dim=8, start=7, end=i64::MIN, step=-1
        let mut node = create_static_shape_slice_node(
            vec![Some(8)],
            vec![7],
            vec![i64::MIN],
            vec![0],
            vec![-1],
        )
        .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                let shape = t.static_shape.as_ref().unwrap();
                assert_eq!(shape[0], Some(8));
            }
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_static_shape_negative_step_typical() {
        // start=7, end=2, step=-1 over dim=10 -> indices 7,6,5,4,3 = 5 elements
        let mut node =
            create_static_shape_slice_node(vec![Some(10)], vec![7], vec![2], vec![0], vec![-1])
                .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.static_shape.as_ref().unwrap()[0], Some(5)),
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_static_shape_negative_indices() {
        // start=-3, end=-1, step=1 over dim=10 -> indices 7,8 = 2 elements
        let mut node =
            create_static_shape_slice_node(vec![Some(10)], vec![-3], vec![-1], vec![0], vec![1])
                .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.static_shape.as_ref().unwrap()[0], Some(2)),
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_static_shape_empty_slice() {
        // start=5, end=5 with positive step yields a zero-length axis.
        let mut node =
            create_static_shape_slice_node(vec![Some(10)], vec![5], vec![5], vec![0], vec![1])
                .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.static_shape.as_ref().unwrap()[0], Some(0)),
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_static_shape_partial_known_preserves_none() {
        // Input has a None for axis 1; slicing only axis 0 must keep axis 1 as None.
        let mut node = create_static_shape_slice_node(
            vec![Some(10), None, Some(8)],
            vec![1],
            vec![6],
            vec![0],
            vec![1],
        )
        .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                let shape = t.static_shape.as_ref().unwrap();
                assert_eq!(shape, &vec![Some(5), None, Some(8)]);
            }
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_tensor_no_static_shape_yields_all_none() {
        // Input has rank but no static_shape; output should be Some(vec![None; rank]).
        let mut node = create_test_node(vec![1], vec![5], Some(vec![0])).build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                let shape = t.static_shape.as_ref().unwrap();
                assert_eq!(shape, &vec![None, None, None]);
            }
            other => panic!("Expected tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_input_negative_step() {
        // Shape slicing with a negative step. Old impl returned 0 here; the
        // refactored normalization correctly returns 3 (indices 4,3,2).
        let mut node = TestNodeBuilder::new(NodeType::Slice, "test_shape_neg_step")
            .input_shape("data", 5)
            .input_tensor_i64_data("starts", vec![4], vec![1])
            .input_tensor_i64_data("ends", vec![1], vec![1])
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![-1], vec![1])
            .output_default("output")
            .build_with_graph_data(16);

        let processor = SliceProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();

        assert!(matches!(&node.outputs[0].ty, ArgType::Shape(3)));
    }

    // TODO: Missing test for mismatched lengths of starts/ends/axes/steps.
    // ONNX spec requires same length but this isn't tested or validated.

    // TODO: Missing test for duplicate axes - e.g., axes=[0, 0] should be invalid.
    // Spec doesn't allow duplicate axes but implementation doesn't validate this.

    // TODO: Missing test for out-of-bounds axes after normalization.
    // E.g., for rank-3 tensor, axes=[5] should be invalid even after negative index handling.

    #[test]
    fn test_slice_scalar_tensor_input() {
        // Slicing a ScalarTensor produces a rank-1 Tensor (slice may change element count)
        let mut node = TestNodeBuilder::new(NodeType::Slice, "test_slice")
            .add_input("data", ArgType::ScalarTensor(DType::I64))
            .input_tensor_i64_data("starts", vec![0], vec![1])
            .input_tensor_i64_data("ends", vec![1], vec![1])
            .output_default("output")
            .build_with_graph_data(16);

        let processor = SliceProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::I64);
                assert_eq!(t.rank, 1);
            }
            other => panic!("Expected Tensor, got {:?}", other),
        }
    }
}
