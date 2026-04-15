//! # HannWindow
//!
//! Generates a Hann window as described in the paper
//! <https://ieeexplore.ieee.org/document/1455106>.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__HannWindow.html>
//!
//! ## Opset Versions
//! - **Opset 17**: Initial version with size input, periodic and output_datatype attributes.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{
    ArgType, Argument, DType, Node, RawNode, RuntimeInputRef, TensorDataExt, TensorType,
};
use crate::node::window_common::resolve_output_dtype;
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, validate_opset,
};

pub use crate::node::window_common::WindowSize;

const OP_NAME: &str = "HannWindow";

/// Configuration for the HannWindow operation.
#[derive(Debug, Clone)]
pub struct HannWindowConfig {
    /// If true, returns a periodic window. If false, returns a symmetric window.
    pub periodic: bool,
    /// The output data type.
    pub output_dtype: DType,
    /// The window size.
    pub size: WindowSize,
}

impl Default for HannWindowConfig {
    fn default() -> Self {
        Self {
            periodic: true,
            output_dtype: DType::F32,
            size: WindowSize::default(),
        }
    }
}

/// Node representation for HannWindow operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct HannWindowNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: HannWindowConfig,
}

pub(crate) struct HannWindowProcessor;

impl NodeProcessor for HannWindowProcessor {
    type Config = HannWindowConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 17,
            max_opset: None,
            inputs: InputSpec::Exact(1),
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
        prefs = prefs.add(&node.inputs[0].name, ArgPreference::ScalarNative);
        Ok(Some(prefs))
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        if !node.inputs.is_empty() && node.inputs[0].is_constant() {
            node.inputs[0].to_static()?;
        }
        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 17)?;

        // Validate input is an integer scalar
        let input = &node.inputs[0];
        let is_valid_scalar = input.ty.is_scalar()
            || matches!(&input.ty, ArgType::Tensor(t) if t.rank == 0)
            || matches!(&input.ty, ArgType::Tensor(t) if t.rank == 1
                && t.static_shape.as_ref().is_some_and(|s| s == &[Some(1)]));
        if !is_valid_scalar {
            return Err(ProcessError::TypeMismatch {
                expected: "scalar input (int32 or int64)".to_string(),
                actual: format!("{:?}", input.ty),
            });
        }

        let input_dtype = input.ty.elem_type();
        if !matches!(input_dtype, DType::I32 | DType::I64) {
            return Err(ProcessError::TypeMismatch {
                expected: "int32 or int64".to_string(),
                actual: format!("{:?}", input_dtype),
            });
        }

        let output_dtype = resolve_output_dtype(node, OP_NAME)?;

        // Extract static shape if size is a constant
        let static_shape = match node.inputs[0].value() {
            Some(data) => {
                let val = data.scalar_i64().map_err(|e| ProcessError::TypeMismatch {
                    expected: "scalar integer for size".to_string(),
                    actual: format!("{e}"),
                })?;
                if val < 0 {
                    return Err(ProcessError::Custom(format!(
                        "{OP_NAME}: size must be non-negative, got {val}"
                    )));
                }
                Some(vec![Some(val as usize)])
            }
            None => None,
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: output_dtype,
            rank: 1,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let periodic = node
            .attrs
            .get("periodic")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(true);

        let output_dtype = resolve_output_dtype(node, OP_NAME)?;

        let size = match node.inputs[0].value() {
            Some(data) => {
                let val = data.scalar_i64().map_err(|e| ProcessError::TypeMismatch {
                    expected: "scalar integer for size".to_string(),
                    actual: format!("{e}"),
                })?;
                if val < 0 {
                    return Err(ProcessError::Custom(format!(
                        "{OP_NAME}: size must be non-negative, got {val}"
                    )));
                }
                WindowSize::Static(val as usize)
            }
            None => WindowSize::Runtime(RuntimeInputRef::new(node.inputs[0].name.clone(), 0)),
        };

        Ok(HannWindowConfig {
            periodic,
            output_dtype,
            size,
        })
    }

    fn build_node(&self, mut builder: RawNode, opset: usize) -> Node {
        let config = self.extract_config(&builder, opset).unwrap_or_else(|e| {
            panic!(
                "{OP_NAME} ({}): config extraction failed: {e}",
                builder.name
            )
        });

        // Drop the size input if static (baked into config).
        if matches!(config.size, WindowSize::Static(_)) {
            builder.inputs.clear();
        }

        Node::HannWindow(HannWindowNode {
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
    use crate::processor::OutputPreferences;

    #[test]
    fn test_hann_window_default() {
        let mut node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_scalar_tensor_i64("size", Some(10))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = HannWindowProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.rank, 1);
                assert_eq!(t.static_shape, Some(vec![Some(10)]));
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_hann_window_double_output() {
        let mut node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_scalar_tensor_i64("size", Some(8))
            .output_tensor_f32("output", 0, None)
            .attr_int("output_datatype", 11) // DOUBLE
            .build_with_graph_data(17);

        let processor = HannWindowProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F64);
                assert_eq!(t.rank, 1);
                assert_eq!(t.static_shape, Some(vec![Some(8)]));
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_hann_window_symmetric() {
        let node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_scalar_tensor_i64("size", Some(10))
            .output_tensor_f32("output", 0, None)
            .attr_int("periodic", 0)
            .build_with_graph_data(17);

        let processor = HannWindowProcessor;
        let config = processor.extract_config(&node, 17).unwrap();
        assert!(!config.periodic);
        assert!(matches!(config.size, WindowSize::Static(10)));
        assert_eq!(config.output_dtype, DType::F32);
    }

    #[test]
    fn test_hann_window_runtime_size() {
        let mut node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_scalar_i64("size")
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = HannWindowProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 1);
                assert_eq!(t.static_shape, None);
            }
            _ => panic!("Expected Tensor output"),
        }

        let config = processor.extract_config(&node, 17).unwrap();
        assert!(matches!(config.size, WindowSize::Runtime(_)));
    }

    #[test]
    fn test_hann_window_i32_input_runtime() {
        // i32 runtime input (not a constant) - exercises the input dtype validation for i32
        let mut node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_tensor_i32("size", 0, None)
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = HannWindowProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.rank, 1);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_hann_window_float_input_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_scalar_f32("size")
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = HannWindowProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(
            matches!(result, Err(ProcessError::TypeMismatch { .. })),
            "Expected type mismatch for float input, got: {result:?}"
        );
    }

    #[test]
    fn test_hann_window_negative_size() {
        let mut node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_scalar_tensor_i64("size", Some(-5))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = HannWindowProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("non-negative")),
            "Expected non-negative error, got: {result:?}"
        );
    }

    #[test]
    fn test_hann_window_integer_output_dtype_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_scalar_tensor_i64("size", Some(10))
            .output_tensor_f32("output", 0, None)
            .attr_int("output_datatype", 7) // INT64
            .build_with_graph_data(17);

        let processor = HannWindowProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(
            matches!(result, Err(ProcessError::InvalidAttribute { .. })),
            "Expected invalid attribute error for integer output dtype, got: {result:?}"
        );
    }

    #[test]
    fn test_hann_window_opset_too_low() {
        let mut node = TestNodeBuilder::new(NodeType::HannWindow, "test_hann")
            .input_scalar_tensor_i64("size", Some(10))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(16);

        let processor = HannWindowProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
    }
}
