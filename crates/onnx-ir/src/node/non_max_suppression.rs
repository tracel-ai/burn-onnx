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
#[derive(Debug, Clone, Default, PartialEq)]
pub enum BoxFormat {
    /// `[y1, x1, y2, x2]`, using any diagonal pair of corners.
    #[default]
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

    let boxes = expect_f32_tensor(boxes, "boxes", 3)?;
    let scores = expect_f32_tensor(scores, "scores", 3)?;

    if let Some(shape) = boxes.static_shape.as_ref()
        && let Some(Some(last_dim)) = shape.get(2)
        && *last_dim != 4
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression input 'boxes' must have shape [batch, num_boxes, 4], got last dimension {last_dim}"
        )));
    }

    validate_matching_dimension(
        boxes.static_shape.as_ref(),
        0,
        scores.static_shape.as_ref(),
        0,
        "batch",
    )?;
    validate_matching_dimension(
        boxes.static_shape.as_ref(),
        1,
        scores.static_shape.as_ref(),
        2,
        "num_boxes",
    )?;

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
    rank: usize,
) -> Result<&'a TensorType, ProcessError> {
    let tensor = match &argument.ty {
        ArgType::Tensor(tensor) => tensor,
        other => {
            return Err(ProcessError::TypeMismatch {
                expected: format!("{name} to be a rank-{rank} float32 tensor"),
                actual: other.to_string(),
            });
        }
    };

    if tensor.rank != rank {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression input '{name}' must be rank {rank}, got rank {}",
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

fn validate_matching_dimension(
    left_shape: Option<&Vec<Option<usize>>>,
    left_index: usize,
    right_shape: Option<&Vec<Option<usize>>>,
    right_index: usize,
    name: &str,
) -> Result<(), ProcessError> {
    let left = left_shape
        .and_then(|shape| shape.get(left_index))
        .copied()
        .flatten();
    let right = right_shape
        .and_then(|shape| shape.get(right_index))
        .copied()
        .flatten();

    if let (Some(left), Some(right)) = (left, right)
        && left != right
    {
        return Err(ProcessError::Custom(format!(
            "NonMaxSuppression inputs 'boxes' and 'scores' must agree on {name}, got {left} and {right}"
        )));
    }

    Ok(())
}

fn validate_scalar_input(
    argument: &Argument,
    name: &str,
    expected_dtype: DType,
) -> Result<(), ProcessError> {
    let actual_dtype = match &argument.ty {
        ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => *dtype,
        ArgType::Tensor(tensor) if tensor.rank == 1 => {
            if let Some(shape) = tensor.static_shape.as_ref()
                && let Some(Some(length)) = shape.first()
                && *length != 1
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

    #[test]
    fn infers_i64_triple_output() {
        let mut node = base_node()
            .input_tensor_i64("max_output_boxes_per_class", 1, Some(vec![1]))
            .input_tensor_f32("iou_threshold", 1, Some(vec![1]))
            .input_tensor_f32("score_threshold", 1, Some(vec![1]))
            .build();

        NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
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
    fn infers_empty_output_when_max_output_is_omitted() {
        let mut node = base_node().build();

        NonMaxSuppressionProcessor
            .infer_types(&mut node, 10, &OutputPreferences::new())
            .unwrap();

        assert_eq!(
            node.outputs[0].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 2,
                static_shape: Some(vec![Some(0), Some(3)]),
            })
        );
    }

    #[test]
    fn infers_empty_output_when_max_output_is_non_positive() {
        let mut node = base_node()
            .input_tensor_i64_data("max_output_boxes_per_class", vec![-1], vec![1])
            .build_with_graph_data(10);

        NonMaxSuppressionProcessor
            .infer_types(&mut node, 10, &OutputPreferences::new())
            .unwrap();

        assert_eq!(
            node.outputs[0].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 2,
                static_shape: Some(vec![Some(0), Some(3)]),
            })
        );
    }

    #[test]
    fn accepts_opset_10_and_rejects_opset_9() {
        let node = base_node().build();
        let spec = NonMaxSuppressionProcessor.spec();

        crate::processor::validate_node_spec(&node, 10, &spec).unwrap();
        let error = crate::processor::validate_node_spec(&node, 9, &spec).unwrap_err();

        assert!(matches!(
            error,
            ProcessError::UnsupportedOpset {
                required: 10,
                actual: 9
            }
        ));
    }

    #[test]
    fn accepts_omitted_middle_input() {
        let mut node = base_node()
            .input_tensor_i64("max_output_boxes_per_class", 1, Some(vec![1]))
            .add_input(
                "",
                ArgType::Tensor(TensorType::new(DType::F32, 1, Some(vec![Some(1)]))),
            )
            .input_tensor_f32("score_threshold", 1, Some(vec![1]))
            .build();

        NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap();
    }

    #[test]
    fn rejects_invalid_center_point_box() {
        let mut node = base_node().attr_int("center_point_box", 2).build();
        let error = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(error, ProcessError::InvalidAttribute { .. }));
    }

    #[test]
    fn rejects_wrong_center_point_box_type() {
        let mut node = base_node().attr_float("center_point_box", 1.0).build();
        let error = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(matches!(error, ProcessError::InvalidAttribute { .. }));
    }

    #[test]
    fn rejects_invalid_boxes_shape() {
        let mut node = TestNodeBuilder::new(NodeType::NonMaxSuppression, "nms")
            .input_tensor_f32("boxes", 3, Some(vec![1, 6, 5]))
            .input_tensor_f32("scores", 3, Some(vec![1, 1, 6]))
            .output_default("selected_indices")
            .build();
        let error = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(error.to_string().contains("last dimension 5"));
    }

    #[test]
    fn rejects_mismatched_box_count() {
        let mut node = TestNodeBuilder::new(NodeType::NonMaxSuppression, "nms")
            .input_tensor_f32("boxes", 3, Some(vec![1, 6, 4]))
            .input_tensor_f32("scores", 3, Some(vec![1, 1, 5]))
            .output_default("selected_indices")
            .build();
        let error = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(error.to_string().contains("num_boxes"));
    }

    #[test]
    fn rejects_non_scalar_optional_input() {
        let mut node = base_node()
            .input_tensor_i64("max_output_boxes_per_class", 1, Some(vec![2]))
            .build();
        let error = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(error.to_string().contains("length 2"));
    }

    #[test]
    fn rejects_out_of_range_constant_iou_threshold() {
        let mut node = base_node()
            .input_tensor_i64("max_output_boxes_per_class", 1, Some(vec![1]))
            .input_tensor_f32_data("iou_threshold", vec![1.5], vec![1])
            .build_with_graph_data(11);
        let error = NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap_err();

        assert!(error.to_string().contains("[0, 1]"));
    }

    #[test]
    fn builds_center_format_node() {
        let mut node = base_node().attr_int("center_point_box", 1).build();
        NonMaxSuppressionProcessor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap();

        let Node::NonMaxSuppression(node) = NonMaxSuppressionProcessor.build_node(node, 11) else {
            panic!("Expected NonMaxSuppression node");
        };

        assert_eq!(node.config.center_point_box, Some(BoxFormat::Center));
    }

    #[test]
    fn preserves_omitted_center_point_box() {
        let mut node = base_node().build();
        NonMaxSuppressionProcessor
            .infer_types(&mut node, 10, &OutputPreferences::new())
            .unwrap();

        let Node::NonMaxSuppression(node) = NonMaxSuppressionProcessor.build_node(node, 10) else {
            panic!("Expected NonMaxSuppression node");
        };

        assert_eq!(node.config.center_point_box, None);
    }
}
