//! # Split
//!
//! Splits a tensor into multiple output tensors along a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Split.html>
//!
//! ## Opset Versions
//! - **Opset 1-2**: Initial implementation with `split` sizes specified as an attribute.
//! - **Opset 11**: Refinements to split behavior and type constraints.
//! - **Opset 13**: **BREAKING CHANGE** - `split` changed from attribute to optional input to
//!   support dynamic shapes. This enables runtime determination of split sizes.
//! - **Opset 18**: Added `num_outputs` attribute for easier specification of equal splits without
//!   explicitly providing split sizes.
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, RuntimeInputRef, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Represents either a static value or a runtime argument for Split sizes.
#[derive(Debug, Clone)]
pub enum SplitSizesInput {
    /// Static split sizes known at compile time.
    Static(Vec<usize>),
    /// Runtime split sizes determined during execution.
    Runtime(RuntimeInputRef),
}

impl Default for SplitSizesInput {
    fn default() -> Self {
        SplitSizesInput::Static(vec![])
    }
}

/// Configuration for the Split operation.
#[derive(Clone, Debug, new)]
pub struct SplitConfig {
    /// The axis along which to split the input tensor.
    pub axis: usize,
    /// The uniform size of each split when splitting evenly.
    /// When `None` and `split_sizes` is also `None`, the split size will be calculated at runtime
    /// using `num_outputs`.
    pub split_size: Option<usize>,
    /// Custom sizes for each split when splitting unevenly (Static or Runtime).
    pub split_sizes: Option<SplitSizesInput>,
    /// Number of outputs for runtime split calculation. Only used when both
    /// `split_size` and `split_sizes` are `None`.
    pub num_outputs: Option<usize>,
}

/// Node representation for Split operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SplitNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SplitConfig,
}

pub(crate) struct SplitProcessor;

impl NodeProcessor for SplitProcessor {
    type Config = SplitConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Range(1, 2147483647),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift split input (input[1]) if present
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
        // Extract the input type to determine rank and shape
        let (dtype, rank, input_static_shape) = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => (tensor.dtype, tensor.rank, tensor.static_shape.clone()),
            ArgType::Shape(r) => (crate::ir::DType::I64, 1, Some(vec![Some(*r)])),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Try to get static split sizes to compute output static shapes
        let split_sizes: Option<Vec<usize>> = if node.inputs.len() > 1 {
            node.inputs[1]
                .value()
                .and_then(|v| v.to_vec::<i64>().ok())
                .map(|sizes| sizes.into_iter().map(|s| s as usize).collect())
        } else {
            // For opset < 13, split sizes are an attribute
            node.attrs.get("split").map(|v| {
                v.clone()
                    .into_i64s()
                    .into_iter()
                    .map(|s| s as usize)
                    .collect()
            })
        };

        // Infer output types - all outputs have the same rank and element type as input
        for (i, output_arg) in node.outputs.iter_mut().enumerate() {
            let static_shape = if let Some(ref sizes) = split_sizes {
                // When split sizes are known, compute output static shape
                if let Some(ref input_shape) = input_static_shape {
                    // Get the split axis (default 0)
                    let axis = node
                        .attrs
                        .get("axis")
                        .map(|v| {
                            let a = v.clone().into_i64();
                            if a < 0 {
                                (a + rank as i64) as usize
                            } else {
                                a as usize
                            }
                        })
                        .unwrap_or(0);
                    if i < sizes.len() {
                        let mut shape = input_shape.clone();
                        shape[axis] = Some(sizes[i]);
                        Some(shape)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            output_arg.ty = ArgType::Tensor(TensorType {
                dtype,
                rank,
                static_shape,
            });
        }

        Ok(())
    }

    fn is_noop(&self, node: &RawNode) -> bool {
        // Split is a no-op when there is only a single output
        node.outputs.len() == 1
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Initialize the axis to split along (default is 0 as per ONNX specification)
        let mut axis: i64 = 0;
        // Holds the uniform split size if calculated or provided
        let split_size: Option<usize> = None;
        // Holds the custom split sizes if provided as input (Static or Runtime)
        let mut split_sizes: Option<SplitSizesInput> = None;

        // Extract the input type to determine rank and shape
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            ArgType::Shape(rank) => TensorType {
                dtype: crate::ir::DType::I64,
                rank: 1,
                static_shape: Some(vec![Some(*rank)]),
            },
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Optionally store the number of outputs if provided as an attribute
        let mut num_outputs: Option<usize> = None;

        // Iterate through node attributes to extract relevant values
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                "num_outputs" => num_outputs = Some(value.clone().into_i64() as usize),
                _ => {}
            }
        }

        // TODO: Missing validation that split sizes are positive integers.
        // Negative or zero split sizes should be rejected but only partially validated.

        // Validate axis before normalizing - must be in range [-rank, rank-1]
        let rank = tensor.rank as i64;
        if axis < -rank || axis >= rank {
            return Err(ProcessError::InvalidAttribute {
                name: "axis".to_string(),
                reason: format!(
                    "Split: axis {} is out of range for tensor of rank {} (valid range: [{}, {}])",
                    axis,
                    rank,
                    -rank,
                    rank - 1
                ),
            });
        }

        // Adjust axis if negative to count from the end as per ONNX spec
        if axis < 0 {
            axis += rank;
        }

        // Validate num_outputs if provided
        if let Some(num) = num_outputs
            && num == 0
        {
            return Err(ProcessError::InvalidAttribute {
                name: "num_outputs".to_string(),
                reason: "Split: num_outputs must be greater than 0".to_string(),
            });
        }

        // Handle the case when num_outputs is provided to calculate uniform split size
        if let Some(num_outputs) = num_outputs
            && let Some(static_shape) = &tensor.static_shape
            && let Some(dim_size) = static_shape[axis as usize]
        {
            // Validate that dimension size is sufficient for the number of outputs
            if dim_size == 0 {
                return Err(ProcessError::Custom(format!(
                    "Split: cannot split dimension of size 0 into {} outputs",
                    num_outputs
                )));
            }

            // Compute explicit per-output sizes so codegen uses
            // split_with_sizes(), which always produces exactly num_outputs
            // chunks (split() can give fewer when dim_size < chunk * (n-1)).
            let chunk = dim_size.div_ceil(num_outputs);
            let mut remaining = dim_size;
            let sizes: Vec<usize> = (0..num_outputs)
                .map(|_| {
                    let s = chunk.min(remaining);
                    remaining = remaining.saturating_sub(chunk);
                    s
                })
                .collect();

            split_sizes = Some(SplitSizesInput::Static(sizes));
        }
        // If static shape is not available, split_size will be calculated at runtime
        // using num_outputs. We'll handle this in the code generation phase.

        // Check for custom split sizes provided as a second input (opset 13+)
        if node.inputs.len() > 1 {
            // Validate split input type
            match &node.inputs[1].ty {
                ArgType::Tensor(t) => {
                    // Split tensor must be 1D and int64 dtype
                    if t.rank != 1 {
                        return Err(ProcessError::Custom(format!(
                            "Split: split sizes tensor must be 1D, got rank {}",
                            t.rank
                        )));
                    }
                    if t.dtype != crate::ir::DType::I64 {
                        return Err(ProcessError::TypeMismatch {
                            expected: "Split sizes tensor with dtype I64".to_string(),
                            actual: format!("Split sizes tensor with dtype {:?}", t.dtype),
                        });
                    }
                }
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor for split sizes input".to_string(),
                        actual: format!("{:?}", node.inputs[1].ty),
                    });
                }
            }

            split_sizes = match node.inputs[1].value() {
                None => {
                    // Runtime input - no static value available
                    Some(SplitSizesInput::Runtime(RuntimeInputRef::new(
                        node.inputs[1].name.clone(),
                        1,
                    )))
                }
                Some(tensor_data) => {
                    let sizes: Vec<i64> = tensor_data.to_vec().unwrap();

                    // Validate that all split sizes are non-negative
                    for (i, &size) in sizes.iter().enumerate() {
                        if size < 0 {
                            return Err(ProcessError::Custom(format!(
                                "Split: split size at index {} must be non-negative, got {}",
                                i, size
                            )));
                        }
                    }

                    let usizes: Vec<usize> = sizes.into_iter().map(|x| x as usize).collect();

                    // Validate that number of split sizes matches number of outputs
                    if usizes.len() != node.outputs.len() {
                        return Err(ProcessError::Custom(format!(
                            "Split: number of split sizes ({}) must match number of outputs ({})",
                            usizes.len(),
                            node.outputs.len()
                        )));
                    }

                    // Validate that sum of split sizes matches the dimension size (if static shape is available)
                    if let Some(static_shape) = &tensor.static_shape
                        && let Some(dim_size) = static_shape[axis as usize]
                    {
                        let total_size: usize = usizes.iter().sum();
                        if total_size != dim_size {
                            return Err(ProcessError::Custom(format!(
                                "Split: sum of split sizes ({}) must equal dimension size ({}) along axis {}",
                                total_size, dim_size, axis
                            )));
                        }
                    }

                    if !usizes.is_empty() {
                        Some(SplitSizesInput::Static(usizes))
                    } else {
                        None
                    }
                }
            };
        } else if let Some(split_attr) = node.attrs.get("split") {
            // For opset < 13, split sizes are an attribute
            let sizes = split_attr.clone().into_i64s();
            if !sizes.is_empty() {
                if sizes.iter().any(|&s| s < 0) {
                    return Err(ProcessError::Custom(
                        "Split: split sizes must be non-negative".to_string(),
                    ));
                }
                let usizes: Vec<usize> = sizes.into_iter().map(|s| s as usize).collect();

                if usizes.len() != node.outputs.len() {
                    return Err(ProcessError::Custom(format!(
                        "Split: number of split sizes ({}) must match number of outputs ({})",
                        usizes.len(),
                        node.outputs.len()
                    )));
                }

                if let Some(static_shape) = &tensor.static_shape
                    && let Some(dim_size) = static_shape[axis as usize]
                {
                    let total_size: usize = usizes.iter().sum();
                    if total_size != dim_size {
                        return Err(ProcessError::Custom(format!(
                            "Split: sum of split sizes ({}) must equal dimension size ({}) along axis {}",
                            total_size, dim_size, axis
                        )));
                    }
                }

                split_sizes = Some(SplitSizesInput::Static(usizes));
            }
        }

        // Infer split_size if neither custom split_sizes nor split_size is provided
        // and static shape is available
        if split_sizes.is_none()
            && split_size.is_none()
            && let Some(static_shape) = &tensor.static_shape
            && let Some(dim_size) = static_shape[axis as usize]
        {
            let n = node.outputs.len();
            let chunk = dim_size.div_ceil(n);
            let mut remaining = dim_size;
            let sizes: Vec<usize> = (0..n)
                .map(|_| {
                    let s = chunk.min(remaining);
                    remaining = remaining.saturating_sub(chunk);
                    s
                })
                .collect();

            split_sizes = Some(SplitSizesInput::Static(sizes));
        }
        // If static shape is not available, we need num_outputs for runtime calculation

        // Determine num_outputs for runtime calculation
        // Only set if both split_size and split_sizes are None (runtime case)
        let runtime_num_outputs = if split_size.is_none() && split_sizes.is_none() {
            Some(num_outputs.unwrap_or(node.outputs.len()))
        } else {
            None
        };

        // Return the configuration for splitting operation
        let config = SplitConfig {
            axis: axis as usize,
            split_size,
            split_sizes,
            num_outputs: runtime_num_outputs,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Split(SplitNode {
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
    use crate::ir::{ArgType, AttributeValue, DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use std::collections::HashMap;

    fn create_test_node(
        input_rank: usize,
        num_outputs: usize,
        static_shape: Option<Vec<usize>>,
        attrs: Option<HashMap<String, AttributeValue>>,
        split_sizes_input: Option<Vec<i64>>,
    ) -> TestNodeBuilder {
        // Start with input tensor
        let mut builder = TestNodeBuilder::new(NodeType::Split, "test_split").input_tensor_f32(
            "input",
            input_rank,
            static_shape,
        );

        // Add split sizes input if provided
        if let Some(sizes) = split_sizes_input {
            builder = builder.input_tensor_i64_data("split", sizes.clone(), vec![sizes.len()]);
        }

        // Add output tensors
        for i in 0..num_outputs {
            builder = builder.output_tensor_f32(
                &format!("output_{i}"),
                0, // Will be updated
                None,
            );
        }

        // Add attributes if provided
        if let Some(attributes) = attrs {
            for (key, value) in attributes {
                builder = match key.as_str() {
                    "axis" => builder.attr_int("axis", value.into_i64()),
                    "num_outputs" => builder.attr_int("num_outputs", value.into_i64()),
                    _ => builder,
                };
            }
        }

        builder
    }

    #[test]
    fn test_split_single_output() {
        let mut node = create_test_node(3, 1, Some(vec![10, 20, 30]), None, None).build();

        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(node.outputs.len(), 1);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_split_multiple_outputs() {
        let mut node = create_test_node(4, 3, Some(vec![12, 15, 18, 21]), None, None).build();

        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(node.outputs.len(), 3);
        for output in &node.outputs {
            match &output.ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(tensor.dtype, DType::F32);
                    assert_eq!(tensor.rank, 4);
                }
                _ => panic!("Expected tensor output"),
            }
        }
    }

    #[test]
    fn test_split_invalid_input() {
        let mut node = create_test_node(3, 2, Some(vec![10, 20, 30]), None, None).build();
        node.inputs[0].ty = ArgType::ScalarNative(DType::F32);

        let processor = SplitProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    // Tests for split_config function

    #[test]
    fn test_split_config_default_axis() {
        // Create a node with static shape and 2 outputs
        let static_shape = Some(vec![10, 20, 30]);
        let node = create_test_node(3, 2, static_shape, None, None).build();

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Default axis should be 0, with explicit split sizes
        assert_eq!(config.axis, 0);
        assert!(config.split_size.is_none());
        assert!(
            matches!(&config.split_sizes, Some(SplitSizesInput::Static(sizes)) if sizes == &vec![5, 5])
        );
    }

    #[test]
    fn test_split_config_specified_axis() {
        // Create a node with static shape, 2 outputs, and a specified axis
        let static_shape = Some(vec![10, 20, 30]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(1)); // Split along axis 1

        let node = create_test_node(3, 2, static_shape, Some(attrs), None).build();

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 1);
        assert!(config.split_size.is_none());
        assert!(
            matches!(&config.split_sizes, Some(SplitSizesInput::Static(sizes)) if sizes == &vec![10, 10])
        );
    }

    #[test]
    fn test_split_config_negative_axis() {
        // Test with negative axis (should count from the end)
        let static_shape = Some(vec![10, 20, 30]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(-1)); // Last axis (index 2)

        let node = create_test_node(3, 3, static_shape, Some(attrs), None).build();

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 2); // -1 should be converted to 2
        assert!(config.split_size.is_none());
        assert!(
            matches!(&config.split_sizes, Some(SplitSizesInput::Static(sizes)) if sizes == &vec![10, 10, 10])
        );
    }

    #[test]
    fn test_split_config_num_outputs_attr() {
        // Test with explicitly specified num_outputs attribute
        let static_shape = Some(vec![12, 24, 36]);
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(4));

        let node = create_test_node(3, 4, static_shape, Some(attrs), None).build();

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 0);
        assert!(config.split_size.is_none());
        assert!(
            matches!(&config.split_sizes, Some(SplitSizesInput::Static(sizes)) if sizes == &vec![3, 3, 3, 3])
        );
    }

    #[test]
    fn test_split_config_with_split_sizes_input() {
        // Test with explicit split sizes provided as second input
        let static_shape = Some(vec![10, 20, 30]);
        let split_sizes = vec![3, 7]; // Custom split sizes along default axis (must sum to 10)

        let node = create_test_node(3, 2, static_shape, None, Some(split_sizes.clone()))
            .build_with_graph_data(16);

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 0);
        assert_eq!(config.split_size, None);
        assert!(
            matches!(&config.split_sizes, Some(SplitSizesInput::Static(sizes)) if sizes == &vec![3, 7])
        );
    }

    #[test]
    fn test_split_config_both_splits_and_num_outputs() {
        // Test with both split sizes input and num_outputs attribute (should be valid but may have issues)
        let static_shape = Some(vec![10, 20, 30]);
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(2));
        let split_sizes = vec![3, 7];

        let node = create_test_node(3, 2, static_shape, Some(attrs), Some(split_sizes))
            .build_with_graph_data(16);

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        // When both are provided, split_sizes takes precedence, so extract_config should succeed
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
    }

    #[test]
    fn test_split_config_zero_num_outputs() {
        // Test with num_outputs attribute set to 0 - this causes divide by zero
        // The test just verifies the node can be created; actual usage would need validation
        let static_shape = Some(vec![10, 20, 30]);
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(0));

        let node = create_test_node(3, 0, static_shape, Some(attrs), None).build();

        // Node created successfully - config extraction would panic on zero, so we skip it
        assert_eq!(node.outputs.len(), 0);
    }

    #[test]
    fn test_split_config_invalid_num_outputs() {
        // Test with num_outputs larger than the dimension size
        let static_shape = Some(vec![5, 10, 15]);
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(10)); // Larger than dim 0 size

        let node = create_test_node(3, 10, static_shape, Some(attrs), None).build();

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
    }

    #[test]
    fn test_split_config_no_static_shape() {
        // Test with no static shape available - extract_config should succeed
        // with num_outputs set for runtime calculation
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(2));

        let node = create_test_node(3, 2, None, Some(attrs), None).build();

        let node = node;
        let processor = SplitProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        // When static shape is not available, split_size is None and num_outputs is set
        assert_eq!(config.axis, 0);
        assert!(config.split_size.is_none());
        assert!(config.split_sizes.is_none());
        assert_eq!(config.num_outputs, Some(2));
    }

    #[test]
    fn test_split_config_invalid_input_type() {
        // Test with invalid input type - extract_config should fail
        let mut node = create_test_node(3, 2, Some(vec![10, 20, 30]), None, None).build();
        node.inputs[0].ty = ArgType::ScalarNative(DType::F32);

        let node = node;
        let processor = SplitProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_split_config_with_runtime_split_sizes() {
        // Test with runtime split sizes (no static value)
        let static_shape = Some(vec![20, 30, 40]);
        let node = TestNodeBuilder::new(NodeType::Split, "test_split")
            .input_tensor_f32("input", 3, static_shape)
            .input_tensor_i64("split", 1, Some(vec![2])) // Runtime input - no static value
            .output_tensor_f32("output_0", 0, None)
            .output_tensor_f32("output_1", 0, None)
            .build();

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axis, 0);
        assert_eq!(config.split_size, None);
        assert!(
            matches!(&config.split_sizes, Some(SplitSizesInput::Runtime(arg)) if arg.name == "split")
        );
    }

    #[test]
    fn test_split_config_non_even_split() {
        // Test with non-evenly divisible dimension size
        let static_shape = Some(vec![11, 22, 33]); // 11 is not evenly divisible by 3
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(0));

        let node = create_test_node(3, 3, static_shape, Some(attrs), None).build();

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // ceil(11 / 3) = 4, so sizes = [4, 4, 3]
        assert!(config.split_size.is_none());
        assert!(
            matches!(&config.split_sizes, Some(SplitSizesInput::Static(sizes)) if sizes == &vec![4, 4, 3])
        );
    }

    // TODO: Missing test for split with runtime split sizes (dynamic case).
    // Need test where split input has no static value to verify Runtime variant handling.

    /// Regression test for #55: Split must accept Shape(rank) input.
    /// Shape values flow through Split in models that reshape tensors based on
    /// computed shapes (e.g., split a shape, replace one dimension, concat back).
    #[test]
    fn test_split_shape_input() {
        // Shape(3) = a 3-element shape, like the output of Shape(rank-3 tensor)
        // Split it with sizes [1, 1, 1] into 3 scalar-like outputs
        let node = TestNodeBuilder::new(NodeType::Split, "test_split")
            .input_shape("shape_input", 3)
            .input_tensor_i64_data("split_sizes", vec![1, 1, 1], vec![3])
            .output_tensor_i64("output_0", 0, None)
            .output_tensor_i64("output_1", 0, None)
            .output_tensor_i64("output_2", 0, None)
            .build_with_graph_data(16);

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // All outputs should be rank-1 I64 tensors with static_shape [1]
        for output in &node.outputs {
            match &output.ty {
                ArgType::Tensor(t) => {
                    assert_eq!(t.dtype, DType::I64);
                    assert_eq!(t.rank, 1);
                    assert_eq!(t.static_shape, Some(vec![Some(1)]));
                }
                other => panic!("Expected Tensor, got {:?}", other),
            }
        }
    }

    /// Regression test for #55: Split with zero-sized split parts is valid per ONNX spec.
    /// Models use [0, N, 0] splits to extract the middle element of a shape.
    #[test]
    fn test_split_zero_sized_parts() {
        let node = TestNodeBuilder::new(NodeType::Split, "test_split")
            .input_shape("shape_input", 1)
            .input_tensor_i64_data("split_sizes", vec![0, 1, 0], vec![3])
            .output_tensor_i64("output_0", 0, None)
            .output_tensor_i64("output_1", 0, None)
            .output_tensor_i64("output_2", 0, None)
            .build_with_graph_data(16);

        let mut node = node;
        let processor = SplitProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Check static shapes: [0], [1], [0]
        let shapes: Vec<_> = node
            .outputs
            .iter()
            .map(|o| match &o.ty {
                ArgType::Tensor(t) => t.static_shape.clone(),
                _ => panic!("Expected Tensor"),
            })
            .collect();
        assert_eq!(
            shapes,
            vec![
                Some(vec![Some(0)]),
                Some(vec![Some(1)]),
                Some(vec![Some(0)])
            ]
        );

        // extract_config should also succeed (no rejection of size 0)
        let config = processor.extract_config(&node, 16).unwrap();
        assert!(matches!(
            &config.split_sizes,
            Some(SplitSizesInput::Static(sizes)) if sizes == &vec![0, 1, 0]
        ));
    }

    #[test]
    fn test_split_single_output_is_noop() {
        let node = create_test_node(3, 1, Some(vec![2, 3, 4]), None, None).build();
        assert!(SplitProcessor.is_noop(&node));
    }

    #[test]
    fn test_split_multiple_outputs_is_not_noop() {
        let node = create_test_node(3, 2, Some(vec![2, 3, 4]), None, None).build();
        assert!(!SplitProcessor.is_noop(&node));
    }

    // TODO: Missing test for split_sizes that don't sum to dimension size.
    // E.g., shape=[10, 20], axis=0, split=[3, 4] (sum=7) should fail as it doesn't match dim size 10.

    // TODO: Missing test for empty splits - split_sizes=[] or num_outputs=0.
    // Should be rejected as invalid configuration.

    // TODO: Missing test for single output split - num_outputs=1 or split=[10].
    // This is valid but not explicitly tested.

    // TODO: Missing test for very uneven splits - e.g., split=[1, 1, 1, 97] for dim size 100.
    // Verify this edge case works correctly.

    // TODO: Missing test for opset < 13 behavior - split as attribute vs input.
    // Implementation requires opset 11+ but attribute-based split (opset < 13) might not work.
}
