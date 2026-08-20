//! # Pad
//!
//! Pads input tensor with additional values at borders.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Pad.html>
//!
//! ## Opset Versions
//! - **Opset 11**: Changed pads from attribute to input for dynamic padding support. Added mode attribute (constant/reflect/edge).
//! - **Opset 13**: Added optional axes input to specify which axes to pad.
//! - **Opset 18**: Added optional constant_value input as alternative to attribute.
//! - **Opset 19**: Added antialiasing support for edge mode (not supported in this implementation).
//!
//! **Implementation Note**: This implementation supports constant, reflect,
//! and edge mode padding on arbitrary dimensions. Static-pads + runtime-axes
//! is rejected (the rare combination would require keeping pads selective
//! through codegen); all other combinations of static/runtime pads and
//! static/runtime axes are supported.
//!
//! TODO: Missing type constraint validation
//! Spec defines type constraints for T (data/output), but implementation doesn't validate.
//! Should validate constant_value type matches data type when provided.
//! Location: extract_config or infer_types
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::ir::{ArgType, AttributeValue, Node, RawNode, RuntimeInputRef, TensorDataExt};

/// Represents either a static value or a runtime argument for pad values.
#[derive(Debug, Clone)]
pub enum PadInput {
    /// Full-rank pads known at compile time as `(before, after)` pairs.
    /// `Vec` length equals the input tensor rank; any `axes` subset has
    /// already been scattered into a full-rank pad list at extraction.
    Static(Vec<(usize, usize)>),
    /// Pads determined at forward time. `input` references
    /// `node.inputs[input_index]`. When `axes` is `Some`, the runtime
    /// values describe only those dimensions and codegen scatters them
    /// into a full-rank pad list whose length is the rank of the Pad
    /// node's data input.
    Runtime {
        input: RuntimeInputRef,
        axes: Option<AxesInput>,
    },
}

/// Opset 18+ `axes` input.
#[derive(Debug, Clone)]
pub enum AxesInput {
    /// Constant axes, already normalized to non-negative in-range indices.
    Static(Vec<usize>),
    /// Runtime axes tensor. Values may include negatives (codegen folds
    /// in `rank + axis` at forward() time).
    Runtime(RuntimeInputRef),
}

/// Represents either a static value or a runtime argument for constant value.
#[derive(Debug, Clone)]
pub enum ConstantValueInput {
    /// Static constant value known at compile time.
    Static(f32),
    /// Runtime constant value determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

/// Padding mode for Pad operation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PadMode {
    /// Constant padding (fill with constant value).
    #[default]
    Constant,
    /// Reflect padding (mirror values).
    Reflect,
    /// Edge padding (replicate edge values).
    Edge,
}

impl std::str::FromStr for PadMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "constant" => Ok(PadMode::Constant),
            "reflect" => Ok(PadMode::Reflect),
            "edge" => Ok(PadMode::Edge),
            _ => Err(format!("Invalid pad mode: {}", s)),
        }
    }
}

impl PadMode {
    /// Convert PadMode to string for serialization.
    pub fn as_str(&self) -> &str {
        match self {
            PadMode::Constant => "constant",
            PadMode::Reflect => "reflect",
            PadMode::Edge => "edge",
        }
    }
}

/// Configuration for the Pad operation.
#[derive(Debug, Clone, new)]
pub struct PadConfig {
    /// The paddings to be applied to each dimension.
    pub pads: PadInput,
    /// The constant value to fill the padded areas with.
    pub constant_value: ConstantValueInput,
    /// The padding mode (constant, reflect, edge). Default: constant.
    pub mode: PadMode,
}

/// Node representation for Pad operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct PadNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: PadConfig,
}

/// Normalize ONNX `axes` values: negative indices become `rank + axis`.
/// Rejects duplicates and out-of-range values with
/// `ProcessError::InvalidAttribute`. A duplicate after normalization
/// (e.g. `[-3, 1]` against rank 4) is detected by the post-normalization
/// contains-check, not by raw-value comparison.
fn normalize_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>, ProcessError> {
    let r = rank as i64;
    let mut out = Vec::with_capacity(axes.len());
    for &a in axes {
        let norm = if a < 0 { a + r } else { a };
        if norm < 0 || norm >= r {
            return Err(ProcessError::InvalidAttribute {
                name: "axes".to_string(),
                reason: format!("axis {a} out of range for rank {rank}"),
            });
        }
        let nu = norm as usize;
        if out.contains(&nu) {
            return Err(ProcessError::InvalidAttribute {
                name: "axes".to_string(),
                reason: format!("duplicate axis {nu}"),
            });
        }
        out.push(nu);
    }
    Ok(out)
}

/// Given per-axis `(before, after)` pad pairs listed in the same order
/// as `axes`, produce a full-rank pads vector where any dimension not
/// in `axes` has `(0, 0)`.
fn expand_axes_pads_to_full(
    pairs: &[(usize, usize)],
    axes: &[usize],
    rank: usize,
) -> Vec<(usize, usize)> {
    let mut full = vec![(0usize, 0usize); rank];
    for (i, &axis) in axes.iter().enumerate() {
        full[axis] = pairs[i];
    }
    full
}

pub(crate) struct PadProcessor;

impl NodeProcessor for PadProcessor {
    type Config = PadConfig;

    fn is_noop(&self, node: &RawNode) -> bool {
        // Pad is a no-op when all pad values are zero (static only)
        // Check pads attribute first ("paddings" in opset 1, "pads" in opset 2+)
        let pads_attr = node
            .attrs
            .get("pads")
            .or_else(|| node.attrs.get("paddings"));
        if let Some(pads_attr) = pads_attr {
            let pads = pads_attr.clone().into_i64s();
            return pads.iter().all(|&p| p == 0);
        }

        // Check pads input (input[1]) if it has static data
        if let Some(input) = node.get_input(1)
            && let Some(tensor_data) = input.value()
            && let Ok(pad_values) = tensor_data.try_into_vec::<i64>()
        {
            return pad_values.iter().all(|&p| p == 0);
        }

        false
    }

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Range(1, 4),
            outputs: OutputSpec::Exact(1),
        }
    }

    // TODO mark axes inputs as Shape if inputs are constant

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift pads input (input[1]) if present and not optional
        if node.inputs.len() > 1 && !node.inputs[1].is_optional() && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        // Lift constant_value input (input[2]) if present and not optional
        if node.inputs.len() > 2 && !node.inputs[2].is_optional() && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        // Lift axes input (input[3]) if present and not optional. Axes
        // is opset 18+; if constant, extract_config will expand the
        // selective pads to full-rank pads.
        if node.inputs.len() > 3 && !node.inputs[3].is_optional() && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add validation for input count (1-4 inputs as per spec)
        // Spec allows 1-4 inputs (data, pads, constant_value optional, axes optional).
        // Currently no explicit validation, though extract_config rejects 4 inputs (axes).
        // Should add: validate_min_inputs(node, 1) and validate_max_inputs(node, 4)
        // Location: After validate_opset

        // TODO: Add validation for output count (should be exactly 1)
        // Missing explicit output count validation. Should add validate_output_count(node, 1).
        // Location: After input count validation

        // TODO: Validate that mode attribute if present is in ["constant", "reflect", "edge"]
        // Mode validation currently only happens in extract_config, not in infer_types.
        // This means type inference succeeds even with invalid mode, failing later.
        // Should validate mode attribute early in infer_types for better error messages.
        // Location: After output count validation

        // Output has same type as input
        if let Some(input) = node.inputs.first() {
            node.outputs[0].ty = input.ty.clone();
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Helper function to get mode
        fn get_mode(node: &RawNode) -> Result<PadMode, ProcessError> {
            use std::str::FromStr;

            // Check for mode attribute (default is "constant")
            for (key, value) in node.attrs.iter() {
                if key.as_str() == "mode" {
                    let mode_str = value.clone().into_string();
                    let mode = PadMode::from_str(&mode_str).map_err(|e| {
                        ProcessError::InvalidAttribute {
                            name: "mode".to_string(),
                            reason: e,
                        }
                    })?;
                    return Ok(mode);
                }
            }
            Ok(PadMode::default())
        }

        fn get_pads(node: &RawNode) -> Result<PadInput, ProcessError> {
            let input_dim = match &node.inputs.first().unwrap().ty {
                ArgType::Tensor(tensor) => tensor.rank,
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor".to_string(),
                        actual: "Pad: Only tensor input is valid".to_string(),
                    });
                }
            };

            // Opset 18+ adds an optional `axes` input at index 3. When
            // present, `pads` describes only those axes (length
            // 2*len(axes)). Static axes is normalized eagerly; runtime
            // axes is deferred to forward() time.
            let axes = match node.get_input(3) {
                None => None,
                Some(input) => match input.value() {
                    None => Some(AxesInput::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        3,
                    ))),
                    Some(tensor_data) => {
                        let raw =
                            tensor_data
                                .to_i64_vec()
                                .map_err(|e| ProcessError::TypeMismatch {
                                    expected: "i64-compatible tensor for axes".to_string(),
                                    actual: e.to_string(),
                                })?;
                        Some(AxesInput::Static(normalize_axes(&raw, input_dim)?))
                    }
                },
            };

            // Resolve the static-side `(expected_pad_pairs, axes_subset)`
            // pair for the static-pads branches, or surface the
            // unsupported static-pads + runtime-axes combination.
            fn static_axes_view(
                axes: &Option<AxesInput>,
                input_dim: usize,
                source: &str,
            ) -> Result<(usize, Option<Vec<usize>>), ProcessError> {
                match axes {
                    Some(AxesInput::Static(a)) => Ok((a.len(), Some(a.clone()))),
                    Some(AxesInput::Runtime(_)) => Err(ProcessError::InvalidAttribute {
                        name: "pads".to_string(),
                        reason: format!(
                            "runtime axes combined with {source} pads is not supported"
                        ),
                    }),
                    None => Ok((input_dim, None)),
                }
            }

            // Check for pads attribute first (takes precedence)
            // "paddings" in opset 1, "pads" in opset 2+
            for (key, value) in node.attrs.iter() {
                if key.as_str() == "pads" || key.as_str() == "paddings" {
                    let flat = parse_i64s_as_usize(&value.clone().into_i64s(), "pads")?;
                    let (expected, static_axes) = static_axes_view(&axes, input_dim, "attribute")?;
                    validate_pads_len_with_axes(&flat, expected, "pads")?;
                    let pairs = onnx_pads_to_pairs(&flat);
                    let full = match static_axes {
                        Some(a) => expand_axes_pads_to_full(&pairs, &a, input_dim),
                        None => pairs,
                    };
                    return Ok(PadInput::Static(full));
                }
            }

            // Check for pads input
            if let Some(input) = node.get_input(1) {
                match input.value() {
                    None => {
                        // Validate length here when statically known
                        // (Shape(N) or Tensor with static_shape) so the
                        // error surfaces at codegen rather than as an
                        // opaque runtime panic. Skipped when axes is
                        // runtime: length depends on a tensor we haven't
                        // seen yet.
                        let expected = match &axes {
                            Some(AxesInput::Static(a)) => Some(a.len()),
                            Some(AxesInput::Runtime(_)) => None,
                            None => Some(input_dim),
                        };
                        if let (Some(static_len), Some(expected)) =
                            (input.ty.first_dim_static_len(), expected)
                        {
                            validate_static_runtime_pads_len(static_len, expected)?;
                        }
                        return Ok(PadInput::Runtime {
                            input: RuntimeInputRef::new(input.name.clone(), 1),
                            axes,
                        });
                    }
                    Some(tensor_data) => {
                        let raw =
                            tensor_data
                                .to_i64_vec()
                                .map_err(|e| ProcessError::TypeMismatch {
                                    expected: "i64-compatible tensor for pads".to_string(),
                                    actual: e.to_string(),
                                })?;
                        let flat = parse_i64s_as_usize(&raw, "pads")?;
                        let (expected, static_axes) = static_axes_view(&axes, input_dim, "static")?;
                        validate_pads_len_with_axes(&flat, expected, "pads")?;
                        let pairs = onnx_pads_to_pairs(&flat);
                        let full = match static_axes {
                            Some(a) => expand_axes_pads_to_full(&pairs, &a, input_dim),
                            None => pairs,
                        };
                        return Ok(PadInput::Static(full));
                    }
                }
            }

            Err(ProcessError::Custom(
                "Pad: pads should be given as attribute or as input".to_string(),
            ))
        }

        fn validate_static_runtime_pads_len(
            actual: usize,
            expected_pairs: usize,
        ) -> Result<(), ProcessError> {
            let expected = expected_pairs * 2;
            if actual != expected {
                return Err(ProcessError::InvalidAttribute {
                    name: "pads".to_string(),
                    reason: format!(
                        "runtime pads length {actual} does not match 2 * num_axes ({expected})"
                    ),
                });
            }
            Ok(())
        }

        /// Parse i64 values as usize, rejecting negatives.
        fn parse_i64s_as_usize(
            values: &[i64],
            attr_name: &str,
        ) -> Result<Vec<usize>, ProcessError> {
            values
                .iter()
                .map(|&x| {
                    if x < 0 {
                        return Err(ProcessError::InvalidAttribute {
                            name: attr_name.to_string(),
                            reason: "Negative pad is not supported".to_string(),
                        });
                    }
                    Ok(x as usize)
                })
                .collect()
        }

        /// Validate that pads length matches 2 * num_axes, where
        /// `num_axes` is either the input rank (no axes input) or the
        /// length of the axes vector.
        fn validate_pads_len_with_axes(
            pads: &[usize],
            num_axes: usize,
            attr_name: &str,
        ) -> Result<(), ProcessError> {
            if pads.len() != num_axes * 2 {
                return Err(ProcessError::InvalidAttribute {
                    name: attr_name.to_string(),
                    reason: "pads should be a 1D tensor of shape [2 * num_axes]".to_string(),
                });
            }
            Ok(())
        }

        /// Convert ONNX flat pads `[begin_d0, begin_d1, ..., end_d0, end_d1, ...]`
        /// to `(before, after)` pairs per dimension.
        fn onnx_pads_to_pairs(flat: &[usize]) -> Vec<(usize, usize)> {
            let n = flat.len() / 2;
            (0..n).map(|i| (flat[i], flat[n + i])).collect()
        }

        fn get_constant_value(node: &RawNode) -> Result<ConstantValueInput, ProcessError> {
            // Check for value attribute first (takes precedence)
            if node.attrs.contains_key("value") {
                let constant_value = node.attrs.get("value").map(|value| match value {
                    AttributeValue::Float32(value) => Ok(*value),
                    _ => Err(ProcessError::InvalidAttribute {
                        name: "value".to_string(),
                        reason: "only float32 values are currently supported for constant value as attribute".to_string(),
                    }),
                }).transpose()?.ok_or_else(|| ProcessError::Custom("constant_value should have had a value".to_string()))?;
                return Ok(ConstantValueInput::Static(constant_value));
            }

            // Check for constant value input
            if let Some(input) = node.get_input(2) {
                match input.value() {
                    None => {
                        // Runtime input - store reference instead of cloning the argument
                        return Ok(ConstantValueInput::Runtime(RuntimeInputRef::new(
                            input.name.clone(),
                            2,
                        )));
                    }
                    Some(tensor_data) => {
                        // TODO: Support int, boolean
                        // Static input - extract the scalar value, converting to f32
                        match tensor_data.scalar_f32() {
                            Ok(value) => return Ok(ConstantValueInput::Static(value)),
                            Err(_) => {
                                return Err(ProcessError::TypeMismatch {
                                    expected: "float value".to_string(),
                                    actual: "only float values are currently supported for constant value".to_string(),
                                });
                            }
                        }
                    }
                }
            }

            // Default to 0.0 if no constant value provided
            Ok(ConstantValueInput::Static(0.0))
        }

        let mode = get_mode(node)?;
        let pads = get_pads(node)?;
        let constant_value = get_constant_value(node)?;

        let config = PadConfig {
            pads,
            constant_value,
            mode,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Pad(PadNode {
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
    use crate::ir::{ArgType, Argument, BoolStore, DType, NodeType, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(
        pad_attrs: Option<Vec<i64>>,
        pad_inputs: Option<Vec<i64>>,
        constant_value_attr: Option<f32>,
        constant_value_input: Option<f32>,
        mode: Option<&str>,
        rank: usize,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("output", rank, None);

        // Add pad inputs if provided
        if let Some(pads) = pad_inputs.clone() {
            let pads_len = pads.len();
            builder = builder.input_tensor_i64_data("pads", pads, vec![pads_len]);
        }

        // Add constant value input if provided
        if let Some(value) = constant_value_input {
            builder = builder.input_scalar_tensor_f32("constant_value", Some(value));
        }

        // Add attributes if provided
        if let Some(pads) = pad_attrs {
            builder = builder.attr_ints("pads", pads);
        }

        if let Some(value) = constant_value_attr {
            builder = builder.attr_float("value", value);
        }

        if let Some(mode_val) = mode {
            builder = builder.attr_string("mode", mode_val);
        }

        builder
    }

    #[test]
    fn test_pad_config_with_attrs() {
        // Test for 2D tensor (rank 2)
        let pads = vec![0, 0, 1, 1];
        let node = create_test_node(
            Some(pads.clone()),
            None,
            Some(0.0),
            None,
            Some("constant"),
            2,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(0, 1), (0, 1)]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
        assert_eq!(config.mode, PadMode::Constant);
    }

    #[test]
    fn test_pad_config_with_inputs() {
        // For a 2D tensor, pads should have 4 values (2*rank)
        let pads = vec![0, 0, 1, 1];
        let node = create_test_node(None, Some(pads.clone()), None, Some(1.0), None, 2)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(0, 1), (0, 1)]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_with_3d_tensor() {
        // For a 3D tensor, pads should have 6 values (2*rank)
        let pads = vec![0, 0, 0, 0, 1, 1];
        let node = create_test_node(
            Some(pads.clone()),
            None,
            Some(0.5),
            None,
            Some("constant"),
            3,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(
            matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(0, 0), (0, 1), (0, 1)])
        );
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.5).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_attrs_override_inputs() {
        // Attributes should override inputs
        let attr_pads = vec![0, 0, 2, 2];
        let input_pads = vec![0, 0, 1, 1];
        let node = create_test_node(
            Some(attr_pads.clone()),
            Some(input_pads),
            Some(0.0),
            Some(1.0),
            Some("constant"),
            2,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(0, 2), (0, 2)]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
    }

    fn create_test_node_with_runtime_inputs() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("pads", 1, None) // Runtime input - no static value
            .input_tensor_f32("constant_value", 0, None) // Runtime input - no static value
            .output_tensor_f32("output", 2, None)
    }

    #[test]
    fn test_pad_config_with_runtime_inputs() {
        let node = create_test_node_with_runtime_inputs().build();
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Check that we have runtime inputs
        assert!(matches!(&config.pads, PadInput::Runtime { input, .. } if input.name == "pads"));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Runtime(arg) if arg.name == "constant_value")
        );
    }

    #[test]
    fn test_pad_config_mixed_static_runtime_pads() {
        // Static pads, runtime constant_value
        let builder = TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("pads", vec![0, 0, 1, 1], vec![4]) // Static
            .input_tensor_f32("constant_value", 0, None) // Runtime
            .output_tensor_f32("output", 2, None);

        let node = builder.build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(0, 1), (0, 1)]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Runtime(arg) if arg.name == "constant_value")
        );
    }

    #[test]
    fn test_pad_config_mixed_runtime_static_constant() {
        // Runtime pads, static constant_value
        let builder = TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("pads", 1, None) // Runtime
            .input_scalar_tensor_f32("constant_value", Some(2.5)) // Static
            .output_tensor_f32("output", 2, None);

        let node = builder.build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(&config.pads, PadInput::Runtime { input, .. } if input.name == "pads"));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 2.5).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_default_constant_value() {
        // Test that constant_value defaults to 0.0 when not provided
        let pads = vec![0, 0, 1, 1];
        let node = create_test_node(None, Some(pads.clone()), None, None, None, 2)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(0, 1), (0, 1)]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_optional_constant_value_defaults_to_zero() {
        // When constant_value input is present but optional (empty name),
        // it should fall through to the default of 0.0
        let builder = TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("pads", vec![0, 0, 1, 1], vec![4])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 0,
                    static_shape: None,
                }),
            )
            .output_tensor_f32("output", 2, None);

        let node = builder.build_with_graph_data(16);
        let processor = PadProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(0, 1), (0, 1)]));
    }

    #[test]
    fn test_pad_config_no_inputs() {
        let mut node = create_test_node(None, None, None, None, None, 2).build_with_graph_data(16);
        node.inputs = vec![];
        let processor = PadProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }

    #[test]
    fn test_pad_config_invalid_input_type() {
        let mut node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, None, 2)
            .build_with_graph_data(16);
        node.inputs[0].ty = ArgType::ScalarNative(DType::F32);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_pad_config_static_pads_with_runtime_axes_rejected() {
        // Static pads + runtime axes is not supported: keeping pads
        // selective through codegen would lose the static-pads
        // representation. Reject explicitly.
        let mut node = create_test_node(None, Some(vec![0, 0, 1, 1]), None, Some(0.0), None, 2)
            .build_with_graph_data(16);
        node.inputs.push(Argument {
            name: "axes".to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 1,
                static_shape: None,
            }),
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });
        let processor = PadProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidAttribute { name, .. }) if name == "pads"
        ));
    }

    #[test]
    fn test_pad_config_runtime_pads_with_runtime_axes() {
        let builder = TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 4, None)
            .input_tensor_i64("pads", 1, None)
            .input_scalar_tensor_f32("value", None)
            .input_tensor_i64("axes", 1, None)
            .output_tensor_f32("output", 4, None);
        let node = builder.build_with_graph_data(18);
        let processor = PadProcessor;
        let config = processor.extract_config(&node, 18).unwrap();
        assert!(matches!(
            &config.pads,
            PadInput::Runtime {
                input,
                axes: Some(AxesInput::Runtime(a)),
            } if input.name == "pads" && a.name == "axes"
        ));
    }

    #[test]
    fn test_pad_config_negative_pad() {
        let node = create_test_node(Some(vec![0, 0, -1, 1]), None, None, None, None, 2)
            .build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_pad_config_reflect_mode() {
        let node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, Some("reflect"), 2)
            .build_with_graph_data(16);
        let processor = PadProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.mode, PadMode::Reflect);
    }

    #[test]
    fn test_pad_config_edge_mode() {
        let node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, Some("edge"), 2)
            .build_with_graph_data(16);
        let processor = PadProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.mode, PadMode::Edge);
    }

    #[test]
    fn test_pad_config_invalid_mode() {
        let node = create_test_node(
            Some(vec![0, 0, 1, 1]),
            None,
            None,
            None,
            Some("invalid_mode"),
            2,
        )
        .build_with_graph_data(16);
        let processor = PadProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_pad_config_no_pads() {
        let node = create_test_node(None, None, None, None, None, 2).build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_pad_config_invalid_pads_length() {
        let node = create_test_node(Some(vec![0, 0, 1]), None, None, None, None, 2)
            .build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_pad_config_1d_tensor() {
        let node =
            create_test_node(Some(vec![1, 2]), None, None, None, None, 1).build_with_graph_data(16);
        let processor = PadProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(1, 2)]));
    }

    #[test]
    fn test_pad_config_all_dimensions() {
        // For a 3D tensor, pad all dimensions including the first
        let node = create_test_node(Some(vec![1, 0, 2, 3, 0, 4]), None, None, None, None, 3)
            .build_with_graph_data(16);
        let processor = PadProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert!(
            matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(1, 3), (0, 0), (2, 4)])
        );
    }

    #[test]
    fn test_pad_config_4d_tensor() {
        // 4D tensor with padding on batch and channel dimensions
        let node = create_test_node(
            Some(vec![1, 0, 2, 3, 2, 0, 4, 5]),
            None,
            None,
            None,
            None,
            4,
        )
        .build_with_graph_data(16);
        let processor = PadProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert!(
            matches!(&config.pads, PadInput::Static(pads) if pads == &vec![(1, 2), (0, 0), (2, 4), (3, 5)])
        );
    }

    #[test]
    fn test_pad_zero_pads_attr_is_noop() {
        let node = create_test_node(Some(vec![0, 0, 0, 0]), None, None, None, None, 2)
            .build_with_graph_data(16);
        let processor = PadProcessor;
        assert!(processor.is_noop(&node));
    }

    #[test]
    fn test_pad_nonzero_pads_attr_is_not_noop() {
        let node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, None, 2)
            .build_with_graph_data(16);
        let processor = PadProcessor;
        assert!(!processor.is_noop(&node));
    }

    #[test]
    fn test_pad_zero_pads_input_is_noop() {
        let node = create_test_node(None, Some(vec![0, 0, 0, 0]), None, None, None, 2)
            .build_with_graph_data(16);
        let processor = PadProcessor;
        assert!(processor.is_noop(&node));
    }

    #[test]
    fn test_pad_nonzero_pads_input_is_not_noop() {
        let node = create_test_node(None, Some(vec![0, 0, 1, 0]), None, None, None, 2)
            .build_with_graph_data(16);
        let processor = PadProcessor;
        assert!(!processor.is_noop(&node));
    }

    #[test]
    fn test_pad_incompatible_pads_dtype_returns_error() {
        use burn_tensor::TensorData;

        // Construct a pads input with bool dtype, which cannot be converted to i64
        let bool_data = TensorData::new(vec![false, false, true, true], vec![4]);
        let node = TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_with_data("pads", DType::Bool(BoolStore::Native), 1, bool_data)
            .output_tensor_f32("output", 2, None)
            .build_with_graph_data(16);
        let processor = PadProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    // ===== normalize_axes =====

    #[test]
    fn normalize_axes_basic() {
        let got = super::normalize_axes(&[0, 2], 4).unwrap();
        assert_eq!(got, vec![0, 2]);
    }

    #[test]
    fn normalize_axes_negative_indices_resolve() {
        let got = super::normalize_axes(&[-1, -2], 4).unwrap();
        assert_eq!(got, vec![3, 2]);
    }

    #[test]
    fn normalize_axes_out_of_range_positive() {
        let err = super::normalize_axes(&[5], 4).unwrap_err();
        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
    }

    #[test]
    fn normalize_axes_out_of_range_negative() {
        let err = super::normalize_axes(&[-5], 4).unwrap_err();
        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
    }

    #[test]
    fn normalize_axes_duplicate_raw() {
        let err = super::normalize_axes(&[1, 1], 4).unwrap_err();
        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
    }

    #[test]
    fn normalize_axes_duplicate_after_normalization() {
        // -3 + 4 == 1, collides with an earlier 1. The check happens
        // post-normalization, so this case is rejected.
        let err = super::normalize_axes(&[1, -3], 4).unwrap_err();
        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
    }

    // ===== expand_axes_pads_to_full =====

    #[test]
    fn expand_axes_pads_to_full_leading() {
        let got = super::expand_axes_pads_to_full(&[(1, 2), (3, 4)], &[0, 1], 4);
        assert_eq!(got, vec![(1, 2), (3, 4), (0, 0), (0, 0)]);
    }

    #[test]
    fn expand_axes_pads_to_full_scattered() {
        // axes=[2, 0] means pair 0 -> dim 2, pair 1 -> dim 0. Verifies the
        // per-axis association, not positional ordering.
        let got = super::expand_axes_pads_to_full(&[(1, 2), (3, 4)], &[2, 0], 4);
        assert_eq!(got, vec![(3, 4), (0, 0), (1, 2), (0, 0)]);
    }

    #[test]
    fn expand_axes_pads_to_full_empty() {
        let got = super::expand_axes_pads_to_full(&[], &[], 3);
        assert_eq!(got, vec![(0, 0), (0, 0), (0, 0)]);
    }
}
