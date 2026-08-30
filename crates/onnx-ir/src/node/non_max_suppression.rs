//! # NonMaxSuppression
//!
//! Filters bounding boxes by score and intersection-over-union overlap.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html>
//!
//! ## Opset Versions
//! - **Opset 10**: Initial version.
//! - **Opset 11**: No semantic changes.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{
    ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorDataExt, TensorType,
};
use crate::processor::{
    ArgPreference, InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences,
    OutputSpec, ProcessError,
};

/// Coordinate representation used by the input boxes.
#[derive(Debug, Clone, PartialEq)]
pub enum BoxFormat {
    /// `[y1, x1, y2, x2]`, using any diagonal pair of corners.
    Corner,
    /// `[x_center, y_center, width, height]`.
    Center,
}

/// Configuration for the ONNX NonMaxSuppression operator.
#[derive(Debug, Clone, Default, new)]
pub struct NonMaxSuppressionConfig {
    /// Coordinate representation selected by the `center_point_box` attribute.
    pub center_point_box: Option<BoxFormat>,
}

/// ONNX NonMaxSuppression node.
#[derive(Debug, Clone, NodeBuilder)]
pub struct NonMaxSuppressionNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: NonMaxSuppressionConfig,
}

pub(crate) struct NonMaxSuppressionProcessor;

impl NodeProcessor for NonMaxSuppressionProcessor {
    type Config = NonMaxSuppressionConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Range(2, 5),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &RawNode,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        let mut preferences = InputPreferences::new();

        for index in 2..=4 {
            if let Some(input) = node.get_input(index) {
                preferences = preferences.add(input.name.clone(), ArgPreference::ScalarNative);
            }
        }

        Ok(Some(preferences))
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        self.extract_config(node, opset)?;
        validate_inputs(node)?;

        let selected_count = match node.get_input(2) {
            None => Some(0),
            Some(input) => input
                .value()
                .and_then(|data| data.scalar_i64().ok())
                .filter(|value| *value <= 0)
                .map(|_| 0),
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank: 2,
            static_shape: Some(vec![selected_count, Some(3)]),
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let center_point_box = match node.attrs.get("center_point_box") {
            None => None,
            Some(AttributeValue::Int64(0)) => Some(BoxFormat::Corner),
            Some(AttributeValue::Int64(1)) => Some(BoxFormat::Center),
            Some(AttributeValue::Int64(value)) => {
                return Err(ProcessError::InvalidAttribute {
                    name: "center_point_box".to_string(),
                    reason: format!("expected 0 or 1, got {value}"),
                });
            }
            Some(value) => {
                return Err(ProcessError::InvalidAttribute {
                    name: "center_point_box".to_string(),
                    reason: format!("expected Int64, got {value:?}"),
                });
            }
        };

        Ok(NonMaxSuppressionConfig { center_point_box })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::NonMaxSuppression(NonMaxSuppressionNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

fn validate_inputs(node: &RawNode) -> Result<(), ProcessError> {
    let boxes = node
        .get_input(0)
        .ok_or_else(|| ProcessError::MissingInput("boxes".to_string()))?;
    let scores = node
        .get_input(1)
        .ok_or_else(|| ProcessError::MissingInput("scores".to_string()))?;

    let boxes = expect_f32_tensor(boxes, "boxes")?;
    let scores = expect_f32_tensor(scores, "scores")?;

    if let Some(shape) = boxes.static_shape.as_ref()
        && let Some(Some(last_dim)) = shape.get(2)
        && *last_dim != 4
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression input 'boxes' must have shape [batch, num_boxes, 4], got last dimension {last_dim}"
        )));
    }

    let known_dim = |shape: Option<&[Option<usize>]>, index: usize| {
        shape.and_then(|shape| shape.get(index)).copied().flatten()
    };
    for (box_index, score_index, name) in [(0, 0, "batch"), (1, 2, "num_boxes")] {
        if let (Some(boxes), Some(scores)) = (
            known_dim(boxes.static_shape.as_deref(), box_index),
            known_dim(scores.static_shape.as_deref(), score_index),
        ) && boxes != scores
        {
            return Err(ProcessError::Custom(format!(
                "NonMaxSuppression inputs 'boxes' and 'scores' must agree on {name}, got {boxes} and {scores}"
            )));
        }
    }

    for (index, name, dtype) in [
        (2, "max_output_boxes_per_class", DType::I64),
        (3, "iou_threshold", DType::F32),
        (4, "score_threshold", DType::F32),
    ] {
        if let Some(input) = node.get_input(index) {
            validate_scalar_input(input, name, dtype)?;
        }
    }

    if let Some(input) = node.get_input(3)
        && let Some(data) = input.value()
        && let Ok(value) = data.scalar_f32()
        && !(0.0..=1.0).contains(&value)
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression iou_threshold must be in [0, 1], got {value}"
        )));
    }

    Ok(())
}

fn expect_f32_tensor<'a>(
    argument: &'a Argument,
    name: &str,
) -> Result<&'a TensorType, ProcessError> {
    let tensor = match &argument.ty {
        ArgType::Tensor(tensor) => tensor,
        other => {
            return Err(ProcessError::TypeMismatch {
                expected: format!("{name} to be a rank-3 float32 tensor"),
                actual: other.to_string(),
            });
        }
    };

    if tensor.rank != 3 {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression input '{name}' must be rank 3, got rank {}",
            tensor.rank
        )));
    }
    if tensor.dtype != DType::F32 {
        return Err(ProcessError::TypeMismatch {
            expected: format!("{name} to have float32 dtype"),
            actual: format!("{:?}", tensor.dtype),
        });
    }

    Ok(tensor)
}

fn validate_scalar_input(
    argument: &Argument,
    name: &str,
    expected_dtype: DType,
) -> Result<(), ProcessError> {
    let actual_dtype = match &argument.ty {
        ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => *dtype,
        ArgType::Tensor(tensor) if tensor.rank == 1 => {
            if let Some(length) = argument.ty.first_dim_static_len()
                && length != 1
            {
                return Err(ProcessError::Custom(format!(
                    "NonMaxSuppression optional input '{name}' must contain one value, got length {length}"
                )));
            }
            tensor.dtype
        }
        other => {
            return Err(ProcessError::TypeMismatch {
                expected: format!("{name} to be a scalar or rank-1 single-value tensor"),
                actual: other.to_string(),
            });
        }
    };

    if actual_dtype != expected_dtype {
        return Err(ProcessError::TypeMismatch {
            expected: format!("{name} to have {expected_dtype:?} dtype"),
            actual: format!("{actual_dtype:?}"),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn base_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::NonMaxSuppression, "nms")
            .input_tensor_f32("boxes", 3, Some(vec![1, 6, 4]))
            .input_tensor_f32("scores", 3, Some(vec![1, 1, 6]))
            .output_default("selected_indices")
    }

    fn infer(mut node: RawNode, opset: usize) -> Result<RawNode, ProcessError> {
        NonMaxSuppressionProcessor.infer_types(&mut node, opset, &OutputPreferences::new())?;
        Ok(node)
    }

    #[test]
    fn infers_i64_triple_output() {
        assert_eq!(NonMaxSuppressionProcessor.spec().min_opset, 10);

        let node = infer(
            base_node()
                .input_tensor_i64("max_output_boxes_per_class", 1, Some(vec![1]))
                .input_tensor_f32("iou_threshold", 1, Some(vec![1]))
                .input_tensor_f32("score_threshold", 1, Some(vec![1]))
                .build(),
            11,
        )
        .unwrap();

        assert_eq!(
            node.outputs[0].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 2,
                static_shape: Some(vec![None, Some(3)]),
            })
        );
    }

    #[test]
    fn infers_empty_output_when_max_output_is_absent_or_non_positive() {
        for (case, node) in [
            ("absent", base_node().build()),
            (
                "non-positive",
                base_node()
                    .input_tensor_i64_data("max_output_boxes_per_class", vec![-1], vec![1])
                    .build_with_graph_data(10),
            ),
        ] {
            let node = infer(node, 10).unwrap();
            let ArgType::Tensor(output) = &node.outputs[0].ty else {
                panic!("Expected tensor output for {case}");
            };
            assert_eq!(output.static_shape, Some(vec![Some(0), Some(3)]), "{case}");
        }
    }

    #[test]
    fn rejects_invalid_center_point_box_attributes() {
        for node in [
            base_node().attr_int("center_point_box", 2).build(),
            base_node().attr_float("center_point_box", 1.0).build(),
        ] {
            assert!(matches!(
                infer(node, 11).unwrap_err(),
                ProcessError::InvalidAttribute { .. }
            ));
        }
    }

    #[test]
    fn rejects_invalid_boxes_shape() {
        let node = TestNodeBuilder::new(NodeType::NonMaxSuppression, "nms")
            .input_tensor_f32("boxes", 3, Some(vec![1, 6, 5]))
            .input_tensor_f32("scores", 3, Some(vec![1, 1, 6]))
            .output_default("selected_indices")
            .build();
        let error = infer(node, 11).unwrap_err();

        assert!(error.to_string().contains("last dimension 5"));
    }

    #[test]
    fn rejects_mismatched_box_count() {
        let node = TestNodeBuilder::new(NodeType::NonMaxSuppression, "nms")
            .input_tensor_f32("boxes", 3, Some(vec![1, 6, 4]))
            .input_tensor_f32("scores", 3, Some(vec![1, 1, 5]))
            .output_default("selected_indices")
            .build();
        let error = infer(node, 11).unwrap_err();

        assert!(error.to_string().contains("num_boxes"));
    }

    #[test]
    fn rejects_non_scalar_optional_input() {
        let node = base_node()
            .input_tensor_i64("max_output_boxes_per_class", 1, Some(vec![2]))
            .build();
        let error = infer(node, 11).unwrap_err();

        assert!(error.to_string().contains("length 2"));
    }

    #[test]
    fn rejects_out_of_range_constant_iou_threshold() {
        let node = base_node()
            .input_tensor_i64("max_output_boxes_per_class", 1, Some(vec![1]))
            .input_tensor_f32_data("iou_threshold", vec![1.5], vec![1])
            .build_with_graph_data(11);
        let error = infer(node, 11).unwrap_err();

        assert!(error.to_string().contains("[0, 1]"));
    }

    #[test]
    fn preserves_omitted_center_point_box() {
        let config = NonMaxSuppressionProcessor
            .extract_config(&base_node().build(), 10)
            .unwrap();

        assert_eq!(config.center_point_box, None);
    }
}
