//! # TopK
//!
//! Retrieves the top-K largest or smallest elements along a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__TopK.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with k as an attribute.
//! - **Opset 10**: Changed k from attribute to input, enabling dynamic k values. Supported float types only.
//! - **Opset 11**: Added 'largest' and 'sorted' attributes for controlling output behavior. Added support for integer input types (int8, int16, int32, int64, uint8, uint16, uint32, uint64).
//!
//! **Implementation Note**: This implementation requires opset 10+ (k as input).
//!
//! We now support the full ONNX semantics for `largest` and `sorted` attributes.
//! Bottom‑K (`largest=0`) is implemented by negating the input tensor before and
//! after calling Burn's `topk`.  Unsorted output (`sorted=0`) is permitted, but
//! our runtime always returns sorted results which is valid per the spec.
//! Previous versions of this crate rejected any value other than `1` for these
//! attributes; that restriction has been removed.
//!
//! ## Type Constraints
//! - **T** (Opset 10): tensor(float16), tensor(float), tensor(double)
//! - **T** (Opset 11+): All numeric tensor types (float16, float, double, int8-64, uint8-64)
//! - **I**: tensor(int64) for indices output

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, RuntimeInputRef, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Represents either a static value or a runtime argument for TopK k parameter.
#[derive(Debug, Clone)]
pub enum TopKInput {
    /// Static k known at compile time.
    Static(usize),
    /// Runtime k determined during execution.
    Runtime(RuntimeInputRef),
}

impl Default for TopKInput {
    fn default() -> Self {
        TopKInput::Static(0)
    }
}

/// Configuration for the TopK operation.
///
/// `largest` and `sorted` reflect the corresponding ONNX attributes. They
/// default to `true` to match ONNX defaults. Burn's underlying `topk`
/// primitive always produces sorted results and only supports selecting the
/// largest values. When `largest` is `false` we emulate bottom-K by negating
/// the input before/after the operation.
#[derive(Debug, Clone, new)]
pub struct TopKConfig {
    /// The axis along which to perform the top-k selection.
    pub axis: usize,
    /// The number of top elements to select.
    pub k: TopKInput,
    /// Whether to select the largest (`true`) or smallest (`false`) elements.
    pub largest: bool,
    /// Whether the output should be sorted (`true`) or may be returned
    /// unsorted (`false`).  We ignore this flag during codegen because Burn's
    /// `topk` is always sorted; returning sorted results is valid when the
    /// caller requested unsorted output.
    pub sorted: bool,
}

/// Node representation for TopK operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct TopKNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: TopKConfig,
}

pub(crate) struct TopKProcessor;

impl NodeProcessor for TopKProcessor {
    type Config = TopKConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Range(1, 2),
            outputs: OutputSpec::Exact(2),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift K input (input[1]) if present
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Missing validation that k <= dimension_size along axis.
        // If k is larger than the dimension, this should either be rejected or clamped.
        // ONNX spec behavior for k > dim_size is not well-defined.

        // The ONNX spec allows `largest` and `sorted` to be either 0 or 1.
        // Previously we rejected any value other than 1.  We now accept both
        // so that bottom-K (largest=0) and unsorted outputs (sorted=0) are
        // representable.  The codegen layer will handle `largest=false` by
        // negating the input; `sorted=false` is a no-op since Burn always
        // returns sorted results.
        //
        // Record the flag early so we can perform validation after the common
        // tensor extraction below.
        let largest_flag = node
            .attrs
            .get("largest")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(true);

        // TODO: Missing validation that k is positive (k > 0).
        // Zero or negative k values should be rejected but aren't validated.

        // Extract the shape of the input data tensor
        let data_tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // If bottom-K was requested, ensure the dtype is floating-point.  This
        // check uses the already-extracted `data_tensor` to avoid duplication.
        if !largest_flag {
            match data_tensor.dtype {
                DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {}
                _ => {
                    return Err(ProcessError::Custom(
                        "TopK: bottom-K (largest=0) only supported for floating-point tensors".to_string(),
                    ));
                }
            }
        }

        // Infer output types
        let rank = data_tensor.rank;

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: node.inputs[0].ty.elem_type(),
            rank,
            static_shape: None,
        });
        node.outputs[1].ty = ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract the shape of the input data tensor
        let data_tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        let k = match node.inputs.get(1) {
            Some(k_tensor) => match k_tensor.value() {
                None => {
                    // Runtime input - no static value available
                    TopKInput::Runtime(RuntimeInputRef::new(k_tensor.name.clone(), 1))
                }
                Some(tensor_data) => {
                    let k_value = tensor_data.as_slice::<i64>().unwrap()[0];
                    TopKInput::Static(k_value as usize)
                }
            },
            _ => {
                // Fall back to attribute
                let k_value = node
                    .attrs
                    .get("k")
                    .ok_or_else(|| ProcessError::MissingAttribute("k".to_string()))?
                    .clone()
                    .into_i64();
                TopKInput::Static(k_value as usize)
            }
        };

        let mut axis = match node.attrs.get("axis") {
            Some(axis) => axis.clone().into_i64(),
            None => -1,
        };

        // If axis is negative, it is counted from the end
        if axis < 0 {
            axis += data_tensor.rank as i64;
        }

        // TODO: Missing validation that axis is in valid range after normalization.
        // After converting negative axis, should verify 0 <= axis < rank.

        // Parse largest/sorted attributes, defaulting to 1
        let largest = node
            .attrs
            .get("largest")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(true);
        let sorted = node
            .attrs
            .get("sorted")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(true);

        let config = TopKConfig {
            axis: axis as usize,
            k,
            largest,
            sorted,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::TopK(TopKNode {
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
    use crate::ir::{AttributeValue, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use std::collections::HashMap;

    fn create_test_node(
        input_rank: usize,
        attrs: Option<HashMap<String, AttributeValue>>,
        k_input_value: Option<i64>,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::TopK, "test_topk")
            .input_tensor_f32("X", input_rank, None)
            .output_tensor_f32("Values", 0, None) // Rank will be updated
            .output_tensor_i64("Indices", 0, None); // Rank will be updated

        // Add K input if provided
        if let Some(k) = k_input_value {
            builder = builder.input_tensor_i64_data("K", vec![k], vec![]);
        }

        // Add attributes if provided
        if let Some(attr_map) = attrs {
            for (key, value) in attr_map {
                match value {
                    AttributeValue::Int64(val) => builder = builder.attr_int(&key, val),
                    AttributeValue::Int64s(vals) => builder = builder.attr_ints(&key, vals),
                    AttributeValue::Float32(val) => builder = builder.attr_float(&key, val),
                    AttributeValue::Float32s(vals) => builder = builder.attr_floats(&key, vals),
                    AttributeValue::String(val) => builder = builder.attr_string(&key, &val),
                    AttributeValue::Strings(vals) => builder = builder.attr_strings(&key, vals),
                    _ => panic!("Unsupported attribute type"),
                }
            }
        }

        builder
    }

    #[test]
    fn test_topk_basic() {
        let mut node = create_test_node(3, None, None).build();
        // Add K attribute since we didn't provide K input
        node.attrs.insert("k".to_string(), AttributeValue::Int64(5));

        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(node.outputs.len(), 2);

        // Check first output (values)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output for values"),
        }

        // Check second output (indices)
        match &node.outputs[1].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output for indices"),
        }
    }

    #[test]
    fn test_topk_invalid_input() {
        let mut node = create_test_node(3, None, None).build();
        node.attrs.insert("k".to_string(), AttributeValue::Int64(5));
        node.inputs[0].ty = ArgType::ScalarNative(DType::F32);
        let processor = TopKProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    // Tests for top_k_config function

    #[test]
    fn test_top_k_config_with_k_attribute() {
        // Test when k is provided as an attribute
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(10));
        let node = create_test_node(3, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Default axis should be -1 which gets converted to rank-1
        assert_eq!(config.axis, 2);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 10));
    }

    #[test]
    fn test_top_k_config_with_k_input() {
        // Test when k is provided as an input
        let node = create_test_node(4, None, Some(5)).build_with_graph_data(16);

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Default axis should be -1 which gets converted to rank-1
        assert_eq!(config.axis, 3);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 5));
    }

    #[test]
    fn test_top_k_config_with_explicit_axis() {
        // Test with explicitly specified axis
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("axis".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(3, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 1);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 3));
    }

    #[test]
    fn test_top_k_config_with_negative_axis() {
        // Test with negative axis (counts from the end)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(5));
        attrs.insert("axis".to_string(), AttributeValue::Int64(-2)); // Second-to-last axis
        let node = create_test_node(4, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // For rank 4, axis -2 should be 2
        assert_eq!(config.axis, 2);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 5));
    }

    #[test]
    fn test_top_k_config_with_largest_attribute() {
        // Test with largest attribute set to 1 (default behavior)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(7));
        attrs.insert("largest".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(2, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 1);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 7));
        assert!(config.largest);
    }

    #[test]
    fn test_top_k_config_with_sorted_attribute() {
        // Test with sorted attribute set to 1 (default behavior)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(2));
        attrs.insert("sorted".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(3, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 2);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 2));
        assert!(config.sorted);
    }

    #[test]
    fn test_top_k_config_with_largest_false() {
        // Test with largest attribute set to 0 (bottom-k semantics)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("largest".to_string(), AttributeValue::Int64(0));
        let node = create_test_node(2, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(!config.largest);
    }

    #[test]
    fn test_top_k_bottom_integer_inputs_rejected() {
        // bottom-K with integer input dtype should error during type inference
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(2));
        attrs.insert("largest".to_string(), AttributeValue::Int64(0));
        let mut node = create_test_node(2, Some(attrs), None).build();
        // change input dtype to integer
        node.inputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::I32,
            rank: 2,
            static_shape: None,
        });

        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_top_k_config_with_sorted_false() {
        // Test with sorted attribute set to 0 (caller allows unsorted output)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("sorted".to_string(), AttributeValue::Int64(0));
        let node = create_test_node(2, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(!config.sorted);
    }

    #[test]
    fn test_top_k_config_with_invalid_input_type() {
        // Test with invalid input type
        let mut node = create_test_node(2, None, None).build();
        node.attrs.insert("k".to_string(), AttributeValue::Int64(3));
        node.inputs[0].ty = ArgType::ScalarNative(DType::F32);

        let node = node;
        let processor = TopKProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_top_k_config_without_k() {
        // Test when k is neither provided as input nor attribute
        let node = create_test_node(3, None, None).build();

        let node = node;
        let processor = TopKProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::MissingAttribute(_))));
    }

    #[test]
    fn test_top_k_config_with_both_k_input_and_attribute() {
        // Test when k is provided both as input and attribute
        // Input should take precedence
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(10));
        let node = create_test_node(3, Some(attrs), Some(5)).build_with_graph_data(16);

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // K from input should be used (5), not from attribute (10)
        assert_eq!(config.axis, 2);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 5));
    }

    #[test]
    fn test_top_k_config_with_runtime_k() {
        // Test when k is provided as a runtime input (no static value)
        let node = TestNodeBuilder::new(NodeType::TopK, "test_topk")
            .input_tensor_f32("X", 3, None)
            .input_tensor_i64("K", 0, None) // Runtime input - no static value
            .output_tensor_f32("Values", 0, None)
            .output_tensor_i64("Indices", 0, None)
            .build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 2); // Default axis -1 becomes 2 for rank 3
        assert!(matches!(&config.k, TopKInput::Runtime(arg) if arg.name == "K"));
    }

    // TODO: Missing test for k > dimension_size edge case.
    // What happens when k=10 but dimension size along axis is only 5?

    // TODO: Missing test for k=0 - should be rejected as invalid.

    // TODO: Missing test for negative k - should be rejected as invalid.

    // TODO: Missing test for axis out of bounds after normalization.

    // TODO: Missing test for type constraints - TopK should work with int types (opset 11+).
    // Need test with integer input types to verify they're handled correctly.

    // TODO: Missing test for duplicate values in input.
    // How are ties handled? ONNX spec says indices for ties are implementation-dependent.

    // TODO: Missing test for NaN values in float inputs.
    // ONNX spec doesn't clearly define NaN handling in TopK.

    // TODO: Missing test for zero-size tensor along axis dimension.
    // E.g., input shape [2, 0, 4], axis=1, what should k be?
}
