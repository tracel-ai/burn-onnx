//! # Clip
//!
//! Clips (limits) the values in the input tensor to a specified min/max range. Values below the minimum
//! are set to the minimum value, and values above the maximum are set to the maximum value.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Clip.html>
//!
//! ## Type Constraints
//!
//! - T: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32),
//!   tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bfloat16)
//!
//! ## Opset Versions
//!
//! - **Opset 1-10**: min and max as float attributes only
//! - **Opset 11**: Changed min and max from attributes to optional inputs (allows runtime values)
//! - **Opset 12**: Extended type support to include integer types (int8-64, uint8-64)
//! - **Opset 13+**: Added bfloat16 support and defined behavior when min > max

use burn_tensor::TensorData;
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, DType, Node, RawNode, RuntimeInputRef, Scalar};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};

/// Represents either a static value or a runtime argument for clip parameters.
#[derive(Debug, Clone)]
pub enum ClipInput {
    /// Static value known at compile time. Integer bounds are kept as
    /// `Scalar::Int`/`Scalar::UInt` so they are not rounded through f64.
    Static(Scalar),
    /// Runtime argument determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

/// Configuration for Clip operation
#[derive(Debug, Clone, new)]
pub struct ClipConfig {
    pub min: Option<ClipInput>,
    pub max: Option<ClipInput>,
}

/// Node representation for Clip operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ClipNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ClipConfig,
}

pub(crate) struct ClipProcessor;

impl NodeProcessor for ClipProcessor {
    type Config = ClipConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift min (input[1]) and max (input[2]) if present and they have constant values
        // For Opset 1-10: min/max are attributes, not inputs (no lifting needed)
        // For Opset 11+: min/max are optional inputs that might be constants or runtime values
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add validation for unexpected attributes
        // TODO: Validate behavior when min > max - spec says "all values are set to max" but not tested
        // TODO: Add test for integer type clipping (int8, int16, int32, uint8, uint16) - opset 12+; int64 and uint32 are covered
        // TODO: Add test for NaN/Inf handling - spec doesn't specify behavior
        // TODO: Validate runtime min/max shapes (opset 11+) - must be scalars; constant bounds are checked in extract_config

        // Infer output type
        same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        /// Read a constant bound in its own numeric kind, so i64/u64 values
        /// above 2^53 stay exact. ONNX requires the bound to share the
        /// input's type; checking the kind here keeps codegen's conversion
        /// to the input's bound type lossless.
        fn static_bound(
            data: &TensorData,
            input_dtype: DType,
            name: &str,
        ) -> Result<Scalar, ProcessError> {
            let dtype = data.dtype();
            let value = if dtype.is_float() && input_dtype.is_float() {
                data.iter::<f64>().next().map(Scalar::Float)
            } else if dtype.is_uint() && input_dtype.is_uint() {
                data.iter::<u64>().next().map(Scalar::UInt)
            } else if dtype.is_int() && input_dtype.is_int() {
                data.iter::<i64>().next().map(Scalar::Int)
            } else {
                return Err(ProcessError::TypeMismatch {
                    expected: format!(
                        "Clip {name} of the same numeric kind as input {input_dtype:?}"
                    ),
                    actual: format!("{dtype:?}"),
                });
            };
            match (value, data.num_elements()) {
                (Some(value), 1) => Ok(value),
                (_, n) => Err(ProcessError::TypeMismatch {
                    expected: format!("scalar Clip {name}"),
                    actual: format!("{n} elements"),
                }),
            }
        }

        fn get_clip_input(
            node: &RawNode,
            index: usize,
            name: &str,
        ) -> Result<Option<ClipInput>, ProcessError> {
            let Some(input) = node.inputs.get(index) else {
                return Ok(None);
            };

            // In ONNX, optional inputs are represented by empty strings
            // Skip optional inputs (those that were never provided)
            if input.is_optional() {
                return Ok(None);
            }

            match input.value() {
                None => {
                    // Runtime input - store reference instead of cloning the argument
                    Ok(Some(ClipInput::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        index,
                    ))))
                }
                Some(tensor_data) => {
                    // Shape inputs carry i64 values but have no DType of their own
                    let input_dtype = match &node.inputs[0].ty {
                        ArgType::Shape(_) => DType::I64,
                        ty => ty.elem_type(),
                    };
                    let value = static_bound(&tensor_data, input_dtype, name)?;
                    Ok(Some(ClipInput::Static(value)))
                }
            }
        }

        let mut min_result: Option<ClipInput> = None;
        let mut max_result: Option<ClipInput> = None;

        // For Clip Opset 1-10, the min and max values are attributes
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "min" => {
                    let min = value.clone().into_f32()? as f64;
                    min_result = Some(ClipInput::Static(Scalar::Float(min)));
                }
                "max" => {
                    let max = value.clone().into_f32()? as f64;
                    max_result = Some(ClipInput::Static(Scalar::Float(max)));
                }
                _ => {}
            }
        }

        // For Clip Opset 11+, the min and max values are inputs
        // Check if inputs are available and attributes weren't set
        if min_result.is_none() {
            min_result = get_clip_input(node, 1, "min")?;
        }

        if max_result.is_none() {
            max_result = get_clip_input(node, 2, "max")?;
        }

        // Neither min nor max specified -> Clip is identity. ONNX
        // explicitly allows this; the codegen handles the "both None"
        // branch as a pass-through.

        let config = ClipConfig {
            min: min_result,
            max: max_result,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::Clip(ClipNode {
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
    use crate::ir::{NodeType, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node_with_attributes(min: Option<f32>, max: Option<f32>) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None);

        if let Some(min_val) = min {
            builder = builder.attr_float("min", min_val);
        }

        if let Some(max_val) = max {
            builder = builder.attr_float("max", max_val);
        }

        builder.build()
    }

    fn create_test_node_with_inputs(min: Option<f32>, max: Option<f32>) -> TestNodeBuilder {
        // In ONNX Clip Opset 11+, inputs are positional:
        // Input 0: input
        // Input 1: min (optional)
        // Input 2: max (optional)
        // We need to maintain the correct positions even if values are None
        TestNodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .input_scalar_tensor_f32("min", min)
            .input_scalar_tensor_f32("max", max)
            .output_tensor_f32("Y", 4, None)
    }

    #[test]
    fn test_clip_config_with_attributes() {
        let node = create_test_node_with_attributes(Some(-1.0), Some(1.0));
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(
            matches!(config.min, Some(ClipInput::Static(Scalar::Float(v))) if (v - (-1.0)).abs() < 1e-6)
        );
        assert!(
            matches!(config.max, Some(ClipInput::Static(Scalar::Float(v))) if (v - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_clip_config_with_attributes_min_only() {
        let node = create_test_node_with_attributes(Some(-1.0), None);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(
            matches!(config.min, Some(ClipInput::Static(Scalar::Float(v))) if (v - (-1.0)).abs() < 1e-6)
        );
        assert!(config.max.is_none());
    }

    #[test]
    fn test_clip_config_with_attributes_max_only() {
        let node = create_test_node_with_attributes(None, Some(1.0));
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(config.min.is_none());
        assert!(
            matches!(config.max, Some(ClipInput::Static(Scalar::Float(v))) if (v - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_clip_config_with_inputs() {
        let node = create_test_node_with_inputs(Some(-1.0), Some(1.0)).build_with_graph_data(16);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(
            matches!(config.min, Some(ClipInput::Static(Scalar::Float(v))) if (v - (-1.0)).abs() < 1e-6)
        );
        assert!(
            matches!(config.max, Some(ClipInput::Static(Scalar::Float(v))) if (v - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_clip_config_with_inputs_min_only() {
        // Note: When None is passed, input_scalar_tensor_f32 creates a runtime input
        // So this test actually has static min and runtime max
        let node = create_test_node_with_inputs(Some(-1.0), None).build_with_graph_data(16);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(
            matches!(config.min, Some(ClipInput::Static(Scalar::Float(v))) if (v - (-1.0)).abs() < 1e-6)
        );
        // max is a runtime input (no static value provided)
        assert!(matches!(config.max, Some(ClipInput::Runtime(_))));
    }

    #[test]
    fn test_clip_config_with_inputs_max_only() {
        // Note: When None is passed, input_scalar_tensor_f32 creates a runtime input
        // So this test actually has runtime min and static max
        let node = create_test_node_with_inputs(None, Some(1.0)).build_with_graph_data(16);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        // min is a runtime input (no static value provided)
        assert!(matches!(config.min, Some(ClipInput::Runtime(_))));
        assert!(
            matches!(config.max, Some(ClipInput::Static(Scalar::Float(v))) if (v - 1.0).abs() < 1e-6)
        );
    }

    fn create_test_node_with_runtime_inputs() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .input_tensor_f32("min", 0, None) // Runtime input - no static value
            .input_tensor_f32("max", 0, None) // Runtime input - no static value
            .output_tensor_f32("Y", 4, None)
    }

    #[test]
    fn test_clip_config_with_runtime_inputs() {
        let node = create_test_node_with_runtime_inputs().build();
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Check that we have runtime inputs
        assert!(matches!(config.min, Some(ClipInput::Runtime(ref arg)) if arg.name == "min"));
        assert!(matches!(config.max, Some(ClipInput::Runtime(ref arg)) if arg.name == "max"));
    }

    #[test]
    fn test_clip_config_mixed_static_runtime() {
        // Static min, runtime max
        let builder = TestNodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .input_scalar_tensor_f32("min", Some(-1.0)) // Static
            .input_tensor_f32("max", 0, None) // Runtime
            .output_tensor_f32("Y", 4, None);

        let node = builder.build_with_graph_data(16);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(
            matches!(config.min, Some(ClipInput::Static(Scalar::Float(v))) if (v - (-1.0)).abs() < 1e-6)
        );
        assert!(matches!(config.max, Some(ClipInput::Runtime(ref arg)) if arg.name == "max"));
    }

    #[test]
    fn test_clip_config_no_min_max() {
        let node = create_test_node_with_attributes(None, None);
        let processor = ClipProcessor;

        // Neither min nor max present -> identity. Extract should succeed
        // and return None for both bounds; codegen handles the pass-through.
        let config = processor.extract_config(&node, 16).unwrap();
        assert!(config.min.is_none());
        assert!(config.max.is_none());
    }

    #[test]
    fn test_clip_lift_constants_with_attributes_only() {
        // Test that lift_constants doesn't try to lift when using attributes (Opset 6-10)
        let mut node = create_test_node_with_attributes(Some(-1.0), Some(1.0));
        let processor = ClipProcessor;

        // This should succeed without error since attributes are not inputs
        processor.lift_constants(&mut node, 16).unwrap();
        // Verify that no inputs were modified (node should have 1 input - the data tensor)
        assert_eq!(node.inputs.len(), 1);
    }

    #[test]
    fn test_clip_lift_constants_with_runtime_inputs() {
        // Test that lift_constants doesn't modify runtime inputs (no constant values)
        let mut node = create_test_node_with_runtime_inputs().build();
        let processor = ClipProcessor;

        // Verify inputs are not constant before lifting
        assert!(!node.inputs[1].is_constant()); // min is Dynamic, not Constant
        assert!(!node.inputs[2].is_constant()); // max is Dynamic, not Constant

        // lift_constants should succeed without modifying non-constant inputs
        processor.lift_constants(&mut node, 16).unwrap();

        // Inputs should remain unchanged (still Dynamic)
        assert!(!node.inputs[1].is_static());
        assert!(!node.inputs[2].is_static());
    }

    #[test]
    fn test_clip_lift_constants_with_static_inputs() {
        // Test that lift_constants converts constant inputs to static
        let mut node =
            create_test_node_with_inputs(Some(-1.0), Some(1.0)).build_with_graph_data(16);
        let processor = ClipProcessor;

        // Verify inputs are constant before lifting
        assert!(node.inputs[1].is_constant()); // min has constant value
        assert!(node.inputs[2].is_constant()); // max has constant value

        // Lift constants - this should convert them to Static
        processor.lift_constants(&mut node, 16).unwrap();

        // Verify inputs were converted to Static
        assert!(node.inputs[1].is_static());
        assert!(node.inputs[2].is_static());
    }

    #[test]
    fn test_clip_config_int64_bound_keeps_precision() {
        // 2^53 + 1 is not representable in f64; it must not be rounded.
        let node = TestNodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_i64("X", 1, None)
            .input_scalar_tensor_i64("min", Some(9_007_199_254_740_993))
            .output_tensor_i64("Y", 1, None)
            .build_with_graph_data(16);

        let config = ClipProcessor.extract_config(&node, 16).unwrap();
        assert!(matches!(
            config.min,
            Some(ClipInput::Static(Scalar::Int(9_007_199_254_740_993)))
        ));
        assert!(config.max.is_none());
    }

    #[test]
    fn test_clip_config_uint64_bounds() {
        // UInt64 bounds used to be dropped because they could not be read as f64.
        let tensor = ArgType::Tensor(TensorType {
            dtype: DType::U64,
            rank: 1,
            static_shape: None,
        });
        let scalar = |v: u64| TensorData::new(vec![v], [0usize; 0]);
        let node = TestNodeBuilder::new(NodeType::Clip, "test_clip")
            .add_input("X", tensor.clone())
            .input_tensor_with_data("min", DType::U64, 0, scalar(3))
            .input_tensor_with_data("max", DType::U64, 0, scalar(u64::MAX - 1))
            .add_output("Y", tensor)
            .build_with_graph_data(16);

        let config = ClipProcessor.extract_config(&node, 16).unwrap();
        assert!(matches!(
            config.min,
            Some(ClipInput::Static(Scalar::UInt(3)))
        ));
        assert!(matches!(
            config.max,
            Some(ClipInput::Static(Scalar::UInt(v))) if v == u64::MAX - 1
        ));
    }

    fn clip_node_with_bound(input_dtype: DType, min: TensorData) -> RawNode {
        let tensor = ArgType::Tensor(TensorType {
            dtype: input_dtype,
            rank: 1,
            static_shape: None,
        });
        let min_dtype = min.dtype();
        let min_rank = min.shape().len();
        TestNodeBuilder::new(NodeType::Clip, "test_clip")
            .add_input("X", tensor.clone())
            .input_tensor_with_data("min", min_dtype, min_rank, min)
            .add_output("Y", tensor)
            .build_with_graph_data(16)
    }

    #[test]
    fn test_clip_config_rejects_bound_kind_mismatch() {
        // A negative int64 bound on a uint8 input has no u64 value to emit.
        let node = clip_node_with_bound(DType::U8, TensorData::new(vec![-1i64], [0usize; 0]));
        let err = ClipProcessor.extract_config(&node, 16).unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }), "{err}");
    }

    #[test]
    fn test_clip_config_rejects_non_scalar_bound() {
        let node = clip_node_with_bound(DType::F32, TensorData::new(vec![0f32, 1.0], [2]));
        let err = ClipProcessor.extract_config(&node, 16).unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }), "{err}");
    }

    #[test]
    fn test_clip_config_bf16_and_int16_bounds() {
        // Neither dtype could be read through scalar_f64, so these bounds used to be dropped.
        let bf16 = half::bf16::from_f32(1.5);
        let node = clip_node_with_bound(DType::BF16, TensorData::new(vec![bf16], [0usize; 0]));
        let config = ClipProcessor.extract_config(&node, 16).unwrap();
        assert!(matches!(
            config.min,
            Some(ClipInput::Static(Scalar::Float(1.5)))
        ));

        let node = clip_node_with_bound(DType::I16, TensorData::new(vec![-300i16], [0usize; 0]));
        let config = ClipProcessor.extract_config(&node, 16).unwrap();
        assert!(matches!(
            config.min,
            Some(ClipInput::Static(Scalar::Int(-300)))
        ));
    }
}
