//! # Multinomial
//!
//! Draws samples from a multinomial (categorical) distribution, given unnormalized
//! log-probabilities for each class.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Multinomial.html>
//!
//! ## Opset Versions
//!
//! - **Opset 7**: Initial version with dtype, sample_size, and seed attributes.
//! - **Opset 22**: Adds bfloat16 to the allowed input types.
//!
//! The `seed` attribute is extracted but cannot be honored: the generated code draws from the
//! backend's random generator.
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::proto_conversion::element_type_from_proto;

/// Configuration for the Multinomial operation
#[derive(Debug, Clone)]
pub struct MultinomialConfig {
    /// Output element type (`dtype` attribute): int32 (default) or int64
    pub dtype: DType,
    /// Number of samples to draw per batch row (`sample_size` attribute, default 1)
    pub sample_size: usize,
    /// Random seed (`seed` attribute). Not honored by the generated code
    pub seed: Option<f32>,
}

impl Default for MultinomialConfig {
    fn default() -> Self {
        Self {
            dtype: DType::I32,
            sample_size: 1,
            seed: None,
        }
    }
}

/// Node representation for the Multinomial operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MultinomialNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: MultinomialConfig,
}

pub(crate) struct MultinomialProcessor;

impl NodeProcessor for MultinomialProcessor {
    type Config = MultinomialConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 7,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", other),
                });
            }
        };

        if tensor.rank != 2 {
            return Err(ProcessError::Custom(format!(
                "Multinomial: input must be 2D [batch_size, class_size], got rank {}",
                tensor.rank
            )));
        }

        if !tensor.dtype.is_float() {
            return Err(ProcessError::TypeMismatch {
                expected: "float tensor (float16, bfloat16, float, double)".to_string(),
                actual: format!("{:?}", tensor.dtype),
            });
        }

        if tensor.dtype == DType::BF16 && opset < 22 {
            return Err(ProcessError::Custom(
                "Multinomial: bfloat16 requires opset 22".into(),
            ));
        }
        if tensor.static_shape.as_ref().and_then(|shape| shape[1]) == Some(0) {
            return Err(ProcessError::Custom(
                "Multinomial: class_size must be positive".into(),
            ));
        }
        let batch = tensor
            .static_shape
            .as_ref()
            .and_then(|shape| shape.first().copied())
            .flatten();

        let config = self.extract_config(node, opset)?;

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: config.dtype,
            rank: 2,
            static_shape: Some(vec![batch, Some(config.sample_size)]),
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut config = MultinomialConfig::default();

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "dtype" => {
                    let code = value.clone().into_i32()?;
                    let dtype = element_type_from_proto(code).map_err(|e| {
                        ProcessError::InvalidAttribute {
                            name: "dtype".to_string(),
                            reason: format!("Multinomial: {e}"),
                        }
                    })?;
                    if !matches!(dtype, DType::I32 | DType::I64) {
                        return Err(ProcessError::InvalidAttribute {
                            name: "dtype".to_string(),
                            reason: format!(
                                "Multinomial output dtype must be int32 or int64, got {dtype:?}"
                            ),
                        });
                    }
                    config.dtype = dtype;
                }
                "sample_size" => {
                    let sample_size = value.clone().into_i64()?;
                    config.sample_size = usize::try_from(sample_size)
                        .ok()
                        .filter(|&n| n > 0)
                        .ok_or_else(|| ProcessError::InvalidAttribute {
                            name: "sample_size".to_string(),
                            reason: format!(
                                "Multinomial sample_size must be at least 1, got {sample_size}"
                            ),
                        })?;
                }
                "seed" => {
                    config.seed = Some(value.clone().into_f32()?);
                }
                _ => {}
            }
        }

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::Multinomial(MultinomialNode {
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
    use crate::protos::tensor_proto::DataType;
    use protobuf::Enum;

    fn builder(static_shape: Option<Vec<usize>>) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Multinomial, "test_multinomial")
            .input_tensor_f32("input", 2, static_shape)
            .output_tensor_i32("output", 0, None)
    }

    fn output_tensor(node: &RawNode) -> &TensorType {
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            other => panic!("Expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_multinomial_defaults() {
        let mut node = builder(Some(vec![3, 5])).build();
        let processor = MultinomialProcessor;
        let config = processor.extract_config(&node, 7).unwrap();
        processor
            .infer_types(&mut node, 7, &OutputPreferences::new())
            .unwrap();

        assert_eq!(config.dtype, DType::I32);
        assert_eq!(config.sample_size, 1);
        assert_eq!(config.seed, None);

        let out = output_tensor(&node);
        assert_eq!(out.dtype, DType::I32);
        assert_eq!(out.rank, 2);
        assert_eq!(out.static_shape, Some(vec![Some(3), Some(1)]));
    }

    #[test]
    fn test_multinomial_attributes() {
        let mut node = builder(None)
            .attr_int("dtype", DataType::INT64.value() as i64)
            .attr_int("sample_size", 7)
            .attr_float("seed", 1.5)
            .build();
        let processor = MultinomialProcessor;
        let config = processor.extract_config(&node, 22).unwrap();
        processor
            .infer_types(&mut node, 22, &OutputPreferences::new())
            .unwrap();

        assert_eq!(config.dtype, DType::I64);
        assert_eq!(config.sample_size, 7);
        assert_eq!(config.seed, Some(1.5));

        let out = output_tensor(&node);
        assert_eq!(out.dtype, DType::I64);
        assert_eq!(out.static_shape, Some(vec![None, Some(7)]));
    }

    #[test]
    fn test_multinomial_f64_input() {
        let mut node = TestNodeBuilder::new(NodeType::Multinomial, "test_multinomial")
            .input_tensor_f64("input", 2, Some(vec![2, 4]))
            .output_tensor_i32("output", 0, None)
            .attr_int("sample_size", 3)
            .build();
        MultinomialProcessor
            .infer_types(&mut node, 7, &OutputPreferences::new())
            .unwrap();
        let out = output_tensor(&node);
        assert_eq!(out.dtype, DType::I32);
        assert_eq!(out.static_shape, Some(vec![Some(2), Some(3)]));
    }

    #[test]
    fn test_multinomial_rejects_float_dtype() {
        let mut node = builder(None)
            .attr_int("dtype", DataType::FLOAT.value() as i64)
            .build();
        let result = MultinomialProcessor.infer_types(&mut node, 7, &OutputPreferences::new());
        assert!(matches!(
            result,
            Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "dtype"
        ));
    }

    #[test]
    fn test_multinomial_rejects_zero_sample_size() {
        let mut node = builder(None).attr_int("sample_size", 0).build();
        let result = MultinomialProcessor.infer_types(&mut node, 7, &OutputPreferences::new());
        assert!(matches!(
            result,
            Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "sample_size"
        ));
    }

    #[test]
    fn test_multinomial_rejects_non_2d_input() {
        let mut node = TestNodeBuilder::new(NodeType::Multinomial, "test_multinomial")
            .input_tensor_f32("input", 3, None)
            .output_tensor_i32("output", 0, None)
            .build();
        let result = MultinomialProcessor.infer_types(&mut node, 7, &OutputPreferences::new());
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_multinomial_rejects_int_input() {
        let mut node = TestNodeBuilder::new(NodeType::Multinomial, "test_multinomial")
            .input_tensor_i64("input", 2, None)
            .output_tensor_i32("output", 0, None)
            .build();
        let result = MultinomialProcessor.infer_types(&mut node, 7, &OutputPreferences::new());
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
    #[test]
    fn rejects_empty_classes_and_negative_sample_count() {
        let mut empty = builder(Some(vec![2, 0])).build();
        assert!(
            MultinomialProcessor
                .infer_types(&mut empty, 7, &OutputPreferences::new())
                .is_err()
        );
        let negative = builder(None).attr_int("sample_size", -1).build();
        assert!(MultinomialProcessor.extract_config(&negative, 7).is_err());
    }

    #[test]
    fn bfloat16_requires_opset_22() {
        let mut node = TestNodeBuilder::new(NodeType::Multinomial, "sample")
            .input_tensor_bf16("input", 2, Some(vec![2, 4]))
            .output_default("output")
            .build();
        assert!(
            MultinomialProcessor
                .infer_types(&mut node, 7, &OutputPreferences::new())
                .is_err()
        );
        MultinomialProcessor
            .infer_types(&mut node, 22, &OutputPreferences::new())
            .unwrap();
    }
}
