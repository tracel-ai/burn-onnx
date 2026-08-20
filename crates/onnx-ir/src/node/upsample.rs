//! # Upsample
//!
//! Scales each dimension of the input tensor by a factor, so that
//! `output_dimension = floor(input_dimension * scale)`.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Upsample.html>
//!
//! ## Opset Versions
//! - **Opset 1**: `height_scale` and `width_scale` float attributes; 4-D input only. Linear mode
//!   is spelled `bilinear` here, not `linear`.
//! - **Opset 7**: `scales` float-list attribute with one entry per input dimension; N-D input.
//! - **Opset 9**: `scales` moved from attribute to input, enabling runtime scales.
//! - **Opset 10**: Deprecated in favor of Resize. No semantic change.
//!
//! **Implementation Note**: Upsample is the operator Resize was created from, and Resize's opset 10
//! semantics are Upsample's: coordinates map asymmetrically and nearest-neighbor rounding floors.
//! This processor therefore extracts Upsample's own attributes and builds a [`ResizeNode`] with
//! those two modes pinned, so the two operators share a single codegen path. Note that Resize's
//! own opset 10 path leaves `nearest_mode` at the opset 11 default rather than pinning floor, so
//! `resize.rs` and this module disagree; the spec is on this module's side.
//!
//! ## Supported subset
//!
//! ONNX allows more than Burn's interpolate can express, so this processor accepts less than the
//! table above suggests:
//! - Rank 3 and rank 4 input only (rank 4 when the scales arrive at runtime).
//! - Batch and channel scales must be 1, and spatial scales must be finite and at least 1.
//! - `nearest` only. `linear` is refused because Burn cannot place samples asymmetrically.
//! - A scale must divide its dimension evenly; see [`validate_nearest_scales`].
use crate::ir::{ArgType, Node, RawNode, RuntimeInputRef, TensorDataExt, TensorType};
use crate::node::resize::{
    CoordinateTransformMode, NearestMode, ResizeConfig, ResizeMode, ResizeNode, ResizeScales,
};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

pub(crate) struct UpsampleProcessor;

/// Drop the batch and channel scales, which Burn's interpolate does not take, and reject the
/// models that would silently lose an upscale by having them dropped.
///
/// `extract_config` checks the input rank before any scales are read, so `scales` is known to
/// have a batch and a channel entry by the time this runs.
///
/// Static scales only. Runtime scales never reach here, and the generated code reads just
/// `scales[2]` and `scales[3]`, so a runtime batch or channel scale is dropped without complaint.
fn spatial_scales(scales: Vec<f32>, input_rank: usize) -> Result<Vec<f32>, ProcessError> {
    if scales.len() != input_rank {
        return Err(ProcessError::Custom(format!(
            "Upsample: scales length ({}) must match input rank ({input_rank})",
            scales.len()
        )));
    }
    if scales[..2].iter().any(|&s| s != 1.0) {
        return Err(ProcessError::Custom(format!(
            "Upsample: scaling the batch or channel dimension is not supported, got scales {:?}",
            &scales[..2]
        )));
    }
    // ONNX: "It takes value greater than or equal to 1." A scale below 1, or a non-finite one,
    // reaches `as usize` in the generated code and saturates to a zero-size dimension.
    if let Some(bad) = scales[2..].iter().find(|s| !s.is_finite() || **s < 1.0) {
        return Err(ProcessError::Custom(format!(
            "Upsample: scales must be finite and greater than or equal to 1, got {bad}"
        )));
    }
    Ok(scales[2..].to_vec())
}

/// Note the representation split: `Static` holds the spatial scales only, with batch and channel
/// already dropped, while `Runtime` refers to the untouched ONNX array and codegen indexes it at
/// `[2]` and `[3]`. Normalizing one to match the other would break the generated code.
fn extract_scales(
    node: &RawNode,
    input_rank: usize,
    opset: usize,
) -> Result<ResizeScales, ProcessError> {
    if opset >= 9 {
        let scales = node
            .get_input(1)
            .ok_or_else(|| ProcessError::MissingInput("scales".to_string()))?;

        return match scales.value() {
            Some(data) => {
                let values = data.to_f32_vec().map_err(|e| {
                    ProcessError::Custom(format!("Upsample: cannot read scales input: {e:?}"))
                })?;
                Ok(ResizeScales::Static(spatial_scales(values, input_rank)?))
            }
            None => Ok(ResizeScales::Runtime(RuntimeInputRef::new(
                scales.name.clone(),
                1,
            ))),
        };
    }

    if opset >= 7 {
        let scales = node
            .attrs
            .get("scales")
            .ok_or_else(|| ProcessError::MissingAttribute("scales".to_string()))?;
        return Ok(ResizeScales::Static(spatial_scales(
            scales.clone().into_f32s(),
            input_rank,
        )?));
    }

    // Opset 1 predates the scales attribute: the spatial scales are two separate attributes and
    // the input is always NCHW.
    if input_rank != 4 {
        return Err(ProcessError::Custom(format!(
            "Upsample: opset 1 requires a 4-D input, got rank {input_rank}"
        )));
    }
    let height_scale = node
        .attrs
        .get("height_scale")
        .ok_or_else(|| ProcessError::MissingAttribute("height_scale".to_string()))?
        .clone()
        .into_f32();
    let width_scale = node
        .attrs
        .get("width_scale")
        .ok_or_else(|| ProcessError::MissingAttribute("width_scale".to_string()))?
        .clone()
        .into_f32();
    Ok(ResizeScales::Static(vec![height_scale, width_scale]))
}

/// Reject the nearest-mode scales Burn cannot reproduce, and warn about the ones we cannot see.
///
/// ONNX picks a source element by scale, `floor(o / scale)`. Burn's nearest kernel picks it by
/// output size, `floor(o * in / out)` where `out = floor(in * scale)`. Those coincide exactly
/// when `in * scale` is a whole number, so an evenly dividing scale is safe and anything else
/// shifts pixels. Burn exposes no way to pass the original scale through, so the divergent cases
/// have to be refused rather than silently imported.
fn validate_nearest_scales(
    node_name: &str,
    scales: &ResizeScales,
    input: &TensorType,
) -> Result<(), ProcessError> {
    let ResizeScales::Static(scales) = scales else {
        // None of the static checks can run against a tensor that only exists at runtime, so name
        // all of what goes unchecked, not just the divergence this function is about.
        log::warn!(
            "Node '{node_name}' (Upsample): scales arrive at runtime and cannot be checked here. \
             The generated code reads only the height and width entries, so a batch or channel \
             scale is ignored rather than rejected, a scale below 1 collapses the dimension, and \
             a scale that does not divide its dimension evenly makes Burn's nearest interpolation \
             select different source elements than the ONNX reference. Tracking: #311"
        );
        return Ok(());
    };

    // Spatial dimensions, aligned with `scales`, which holds spatial entries only.
    let dims: Vec<Option<usize>> = match &input.static_shape {
        Some(shape) => shape.iter().skip(2).copied().collect(),
        None => vec![None; scales.len()],
    };

    // Dimensions we know: prove the product is whole, or refuse.
    for (spatial, dim) in dims.iter().enumerate() {
        let (Some(dim), scale) = (dim, scales[spatial]) else {
            continue;
        };
        let scaled = *dim as f64 * scale as f64;
        if scaled.fract() != 0.0 {
            return Err(ProcessError::Custom(format!(
                "Upsample: scale {scale} on dim {} (size {dim}) yields {scaled}, and Burn's \
                 nearest interpolation indexes by output size rather than by scale, so output \
                 would differ from the ONNX reference. Tracking: #311",
                spatial + 2
            )));
        }
    }

    // Dimensions we do not know: an integral scale is safe for every possible size, anything
    // else might not be.
    let unprovable = dims
        .iter()
        .zip(scales)
        .any(|(dim, scale)| dim.is_none() && scale.fract() != 0.0);

    if unprovable {
        log::warn!(
            "Node '{node_name}' (Upsample): scales {scales:?} are not whole numbers and the \
             matching dimensions are dynamic. Where a scale does not divide its dimension evenly, \
             Burn's nearest interpolation selects different source elements than the ONNX \
             reference. Tracking: #311"
        );
    }

    Ok(())
}

fn input_tensor(node: &RawNode) -> Result<&TensorType, ProcessError> {
    match &node
        .inputs
        .first()
        .ok_or_else(|| ProcessError::MissingInput("input".to_string()))?
        .ty
    {
        ArgType::Tensor(tensor) => Ok(tensor),
        other => Err(ProcessError::TypeMismatch {
            expected: "Tensor".to_string(),
            actual: format!("{other:?}"),
        }),
    }
}

impl NodeProcessor for UpsampleProcessor {
    type Config = ResizeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Range(1, 2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift scales input (input[1], opset 9+) if present and constant
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
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
        let config = self.extract_config(node, opset)?;
        let input = input_tensor(node)?.clone();

        // output_dimension = floor(input_dimension * scale), for the dimensions we know
        // statically. `extract_config` has already refused any product that is not whole, so the
        // flooring here never actually discards anything; it is what the spec says the size is.
        let static_shape = match (&config.scales, &input.static_shape) {
            (Some(ResizeScales::Static(scales)), Some(shape)) => Some(
                shape
                    .iter()
                    .enumerate()
                    .map(|(axis, dim)| match (dim, axis.checked_sub(2)) {
                        // f64 to match `calculate_output_size` in burn-nn, which the generated
                        // code goes through.
                        (Some(dim), Some(spatial)) => {
                            Some((*dim as f64 * scales[spatial] as f64).floor() as usize)
                        }
                        (dim, _) => *dim,
                    })
                    .collect(),
            ),
            _ => None,
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: input.dtype,
            rank: input.rank,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, opset: usize) -> Result<Self::Config, ProcessError> {
        let input = input_tensor(node)?.clone();
        let input_rank = input.rank;

        // Burn's interpolate covers 1-D and 2-D spatial input. Checking before reading the scales
        // also guarantees there is a batch and a channel entry for `spatial_scales` to drop.
        if !(3..=4).contains(&input_rank) {
            return Err(ProcessError::Custom(format!(
                "Upsample: only rank 3 and rank 4 inputs are supported, got rank {input_rank}"
            )));
        }

        let mode = match node.attrs.get("mode") {
            // Opset 1 spells linear interpolation "bilinear"; opset 7 renamed it "linear" while
            // sanctioning "bilinear"/"trilinear" as spellings of the same mode.
            Some(mode) => match mode.clone().into_string().to_lowercase().as_str() {
                "bilinear" | "trilinear" => ResizeMode::Linear,
                other => {
                    other
                        .parse::<ResizeMode>()
                        .map_err(|e| ProcessError::InvalidAttribute {
                            name: "mode".to_string(),
                            reason: e,
                        })?
                }
            },
            None => ResizeMode::Nearest,
        };
        match mode {
            ResizeMode::Nearest => {}
            // Burn's bilinear interpolation places samples at half-pixel coordinates
            // (`(o + 0.5) / scale - 0.5`), which is ONNX's `half_pixel` mode. Upsample mandates
            // `asymmetric` (`o / scale`), and Burn exposes no way to ask for it, so every interior
            // sample would be drawn from the wrong place.
            ResizeMode::Linear => {
                return Err(ProcessError::InvalidAttribute {
                    name: "mode".to_string(),
                    reason: "Upsample mode 'linear' requires ONNX's asymmetric coordinate \
                             mapping, which Burn's bilinear interpolation does not implement; \
                             results would differ from the reference. Tracking: #311"
                        .to_string(),
                });
            }
            ResizeMode::Cubic => {
                return Err(ProcessError::InvalidAttribute {
                    name: "mode".to_string(),
                    reason: "Upsample defines only 'nearest' and 'linear'".to_string(),
                });
            }
        }

        let scales = extract_scales(node, input_rank, opset)?;

        // The runtime form indexes `input_dims[3]` on a `[usize; 3]`, which rustc rejects under
        // the deny-by-default `unconditional_panic` lint, so rank 3 would not even compile.
        if matches!(scales, ResizeScales::Runtime(_)) && input_rank != 4 {
            return Err(ProcessError::Custom(format!(
                "Upsample: runtime scales require a rank 4 input, got rank {input_rank}"
            )));
        }

        validate_nearest_scales(&node.name, &scales, &input)?;

        Ok(ResizeConfig {
            mode,
            scales: Some(scales),
            sizes: None,
            // Upsample's coordinate mapping is Resize's asymmetric mode, and its nearest-neighbor
            // rule (input index = floor(output index / scale)) is Resize's floor mode. Both are
            // recorded for fidelity: codegen reads neither, deriving only `align_corners` from
            // the coordinate mode, which is why `validate_nearest_scales` has to exist.
            coordinate_transformation_mode: CoordinateTransformMode::Asymmetric,
            nearest_mode: NearestMode::Floor,
            ..Default::default()
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        // `lift_constants` runs a second time after identity elimination (post_processing.rs),
        // and type inference does not run again after it. So `Constant -> Identity -> Upsample`
        // arrives here with scales that were Runtime during `infer_types` and are Static now,
        // reaching the scale checks for the first time at a point that cannot return an error.
        // Rejecting late beats importing a model we would compute wrong, so this panics with the
        // reason rather than degrading to a warning.
        let config = self
            .extract_config(&builder, opset)
            .unwrap_or_else(|e| panic!("Node '{}' (Upsample): {e}", builder.name));

        Node::Upsample(ResizeNode {
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

    fn static_scales(config: &ResizeConfig) -> Vec<f32> {
        match &config.scales {
            Some(ResizeScales::Static(scales)) => scales.clone(),
            other => panic!("Expected static scales, got {other:?}"),
        }
    }

    #[test]
    fn opset1_height_and_width_attributes() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, Some(vec![1, 3, 10, 20]))
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .attr_float("height_scale", 2.0)
            .attr_float("width_scale", 3.0)
            .build();

        let config = UpsampleProcessor.extract_config(&node, 1).unwrap();

        assert_eq!(config.mode, ResizeMode::Nearest);
        assert_eq!(static_scales(&config), vec![2.0, 3.0]);
        assert!(config.sizes.is_none());
        assert_eq!(
            config.coordinate_transformation_mode,
            CoordinateTransformMode::Asymmetric
        );
        assert_eq!(config.nearest_mode, NearestMode::Floor);
    }

    #[test]
    fn opset1_rejects_non_4d_input() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .attr_float("height_scale", 2.0)
            .attr_float("width_scale", 2.0)
            .build();

        let err = UpsampleProcessor.extract_config(&node, 1).unwrap_err();

        assert!(
            err.to_string().contains("opset 1 requires a 4-D input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn opset7_scales_attribute() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .attr_floats("scales", vec![1.0, 1.0, 2.0, 3.0])
            .build();

        let config = UpsampleProcessor.extract_config(&node, 7).unwrap();

        assert_eq!(config.mode, ResizeMode::Nearest);
        assert_eq!(static_scales(&config), vec![2.0, 3.0]);
    }

    #[test]
    fn rejects_linear_mode() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "linear")
            .attr_floats("scales", vec![1.0, 1.0, 2.0, 2.0])
            .build();

        let err = UpsampleProcessor.extract_config(&node, 7).unwrap_err();

        assert!(
            err.to_string().contains("asymmetric coordinate mapping"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn opset1_bilinear_is_linear_not_an_unknown_mode() {
        // Opset 1 spells linear "bilinear". It is refused for the same reason as "linear", but
        // the reason must say so rather than claiming the mode is unrecognized.
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "bilinear")
            .attr_float("height_scale", 2.0)
            .attr_float("width_scale", 2.0)
            .build();

        let err = UpsampleProcessor.extract_config(&node, 1).unwrap_err();

        assert!(
            err.to_string().contains("asymmetric coordinate mapping"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_scale_that_does_not_divide_its_dimension() {
        // Burn's nearest kernel indexes by output size, so a scale that leaves a fractional
        // product selects different source elements than ONNX does.
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, Some(vec![1, 1, 5, 5]))
            .output_tensor_f32("Y", 4, None)
            .attr_floats("scales", vec![1.0, 1.0, 1.0, 1.75])
            .build();

        let err = UpsampleProcessor.extract_config(&node, 7).unwrap_err();

        assert!(
            err.to_string()
                .contains("scale 1.75 on dim 3 (size 5) yields 8.75"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn accepts_fractional_scale_that_divides_evenly() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, Some(vec![1, 1, 4, 6]))
            .output_tensor_f32("Y", 4, None)
            .attr_floats("scales", vec![1.0, 1.0, 1.5, 1.5])
            .build();

        let config = UpsampleProcessor.extract_config(&node, 7).unwrap();

        assert_eq!(static_scales(&config), vec![1.5, 1.5]);
    }

    #[test]
    fn rejects_scale_below_one() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_floats("scales", vec![1.0, 1.0, 0.5, 0.5])
            .build();

        let err = UpsampleProcessor.extract_config(&node, 7).unwrap_err();

        assert!(
            err.to_string()
                .contains("must be finite and greater than or equal to 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn opset7_defaults_to_nearest() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .attr_floats("scales", vec![1.0, 1.0, 4.0])
            .build();

        let config = UpsampleProcessor.extract_config(&node, 7).unwrap();

        assert_eq!(config.mode, ResizeMode::Nearest);
        assert_eq!(static_scales(&config), vec![4.0]);
    }

    #[test]
    fn opset7_rejects_scales_length_mismatch() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_floats("scales", vec![1.0, 2.0, 2.0])
            .build();

        let err = UpsampleProcessor.extract_config(&node, 7).unwrap_err();

        assert!(
            err.to_string()
                .contains("scales length (3) must match input rank (4)"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_batch_or_channel_scaling() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_floats("scales", vec![1.0, 2.0, 2.0, 2.0])
            .build();

        let err = UpsampleProcessor.extract_config(&node, 7).unwrap_err();

        assert!(
            err.to_string()
                .contains("scaling the batch or channel dimension is not supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn opset9_static_scales_input() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .input_tensor_f32_data("scales", vec![1.0, 1.0, 2.0, 3.0], vec![4])
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .build_with_graph_data(9);

        let config = UpsampleProcessor.extract_config(&node, 9).unwrap();

        assert_eq!(static_scales(&config), vec![2.0, 3.0]);
    }

    #[test]
    fn opset9_runtime_scales_input() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .input_tensor_f32("scales", 1, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .build();

        let config = UpsampleProcessor.extract_config(&node, 9).unwrap();

        match &config.scales {
            Some(ResizeScales::Runtime(reference)) => {
                assert_eq!(reference.name, "scales");
                assert_eq!(reference.input_index, 1);
            }
            other => panic!("Expected runtime scales, got {other:?}"),
        }
    }

    #[test]
    fn rejects_cubic_mode() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "cubic")
            .attr_floats("scales", vec![1.0, 1.0, 2.0, 2.0])
            .build();

        let err = UpsampleProcessor.extract_config(&node, 7).unwrap_err();

        assert!(
            err.to_string()
                .contains("Upsample defines only 'nearest' and 'linear'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn infers_scaled_static_shape() {
        let mut node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, Some(vec![1, 3, 10, 20]))
            .output_tensor_f32("Y", 4, None)
            .attr_floats("scales", vec![1.0, 1.0, 2.0, 1.5])
            .build();

        UpsampleProcessor
            .infer_types(&mut node, 7, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 4);
                assert_eq!(
                    tensor.static_shape,
                    Some(vec![Some(1), Some(3), Some(20), Some(30)])
                );
            }
            other => panic!("Expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn runtime_scales_leave_shape_unknown() {
        let mut node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, Some(vec![1, 3, 10, 20]))
            .input_tensor_f32("scales", 1, None)
            .output_tensor_f32("Y", 4, None)
            .build();

        UpsampleProcessor
            .infer_types(&mut node, 9, &OutputPreferences::new())
            .unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 4);
                assert_eq!(tensor.static_shape, None);
            }
            other => panic!("Expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn rejects_rank_5_input() {
        let mut node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 5, None)
            .output_tensor_f32("Y", 5, None)
            .attr_floats("scales", vec![1.0, 1.0, 2.0, 2.0, 2.0])
            .build();

        let err = UpsampleProcessor
            .infer_types(&mut node, 7, &OutputPreferences::new())
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("only rank 3 and rank 4 inputs are supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_rank_1_input_without_indexing_past_the_scales() {
        // Upsample of a 1-D tensor is legal ONNX, and its scales list has no channel entry for
        // `spatial_scales` to inspect, so the rank check has to come first.
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 1, None)
            .output_tensor_f32("Y", 1, None)
            .attr_floats("scales", vec![2.0])
            .build();

        let err = UpsampleProcessor.extract_config(&node, 7).unwrap_err();

        assert!(
            err.to_string()
                .contains("only rank 3 and rank 4 inputs are supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn builds_the_upsample_variant_not_the_resize_one() {
        // `Node::Upsample` and `Node::Resize` wrap the same struct, so writing the wrong variant
        // here would compile. Only `node_type()` can tell them apart, and hook dispatch keys on
        // it, so pin it: the Display impl cannot, since it prints the inner struct's name.
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_floats("scales", vec![1.0, 1.0, 2.0, 2.0])
            .build();

        let built = UpsampleProcessor.build_node(node, 7);

        assert_eq!(built.node_type(), NodeType::Upsample);
    }

    #[test]
    fn rejects_runtime_scales_on_rank_3_input() {
        let node = TestNodeBuilder::new(NodeType::Upsample, "test_upsample")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32("scales", 1, None)
            .output_tensor_f32("Y", 3, None)
            .build();

        let err = UpsampleProcessor.extract_config(&node, 9).unwrap_err();

        assert!(
            err.to_string()
                .contains("runtime scales require a rank 4 input"),
            "unexpected error: {err}"
        );
    }
}
