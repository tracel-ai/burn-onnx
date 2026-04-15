//! # Mod
//!
//! Element-wise binary modulus operation with Numpy-style broadcasting support.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Mod.html>
//!
//! ## Opset Versions
//! - **Opset 10-12**: Initial implementation with fmod attribute
//! - **Opset 13+**: Extended type support (added bfloat16)
//!
//! ## Missing Test Coverage
//! - TODO: No test for fmod values other than 0 or 1 - Spec only defines 0 and 1, other values should be rejected
//! - TODO: No test for dtype validation - Should ensure both inputs have compatible numeric types
//! - TODO: No test for zero divisor - Division by zero handling not tested
//! - TODO: No test for negative divisors with both fmod modes - Sign handling edge cases
//! - TODO: No test for integer types - Spec supports int8, int16, int32, int64, uint8, uint16, uint32, uint64
//! - TODO: No test for mixed sign operands - fmod=0 vs fmod=1 produces different results

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, Node, RawNode};
use crate::processor::{
    InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec,
    ProcessError,
};

/// Configuration for Mod operations
#[derive(Debug, Clone)]
pub struct ModConfig {
    /// Determines the modulo operation behavior:
    /// false (default): Integer modulo - sign follows divisor (Python-style %)
    /// true: Floating-point modulo (C-style fmod) - sign follows dividend
    pub fmod: bool,
}

/// Node representation for Mod operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ModNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ModConfig,
}

impl ModConfig {
    /// Create a new ModConfig
    pub fn new(fmod: bool) -> Self {
        Self { fmod }
    }
}

pub(crate) struct ModuloProcessor;

impl NodeProcessor for ModuloProcessor {
    type Config = ModConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &RawNode,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        use crate::processor::ArgPreference;

        if node.inputs.len() != 2 {
            return Ok(None);
        }

        let mut prefs = InputPreferences::new();

        // Type propagation for Shape arithmetic (same as Add/Sub/Mul/Div)
        // Case 1: Shape op Constant => prefer Constant as Shape (or ScalarNative for scalars)
        if node.inputs[0].ty.is_shape() {
            if node.inputs[1].ty.is_scalar() {
                prefs = prefs.add(&node.inputs[1].name, ArgPreference::ScalarNative);
            } else {
                prefs = prefs.add(&node.inputs[1].name, ArgPreference::Shape);
            }
        }

        // Case 2: Constant op Shape => prefer Constant as Shape (or ScalarNative for scalars)
        if node.inputs[1].ty.is_shape() {
            if node.inputs[0].ty.is_scalar() {
                prefs = prefs.add(&node.inputs[0].name, ArgPreference::ScalarNative);
            } else {
                prefs = prefs.add(&node.inputs[0].name, ArgPreference::Shape);
            }
        }

        // Type propagation for ScalarNative arithmetic
        if node.inputs[0].ty.is_scalar_native() {
            prefs = prefs.add(&node.inputs[1].name, ArgPreference::ScalarNative);
        }

        if node.inputs[1].ty.is_scalar_native() {
            prefs = prefs.add(&node.inputs[0].name, ArgPreference::ScalarNative);
        }

        Ok(Some(prefs))
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input dtypes are numeric - Integer and floating-point types supported - burn/crates/onnx-ir/src/node/modulo.rs:100
        // TODO: Validate both inputs have same dtype - Mixed types should be rejected - burn/crates/onnx-ir/src/node/modulo.rs:100
        // TODO: Add validation that fmod attribute, if present, is either 0 or 1 - Other values are undefined - burn/crates/onnx-ir/src/node/modulo.rs:100

        // Structural validation for Shape combined with an on-device tensor.
        // burn-onnx codegen for these arms always materializes the Shape as
        // a rank-1 Int tensor, so it can only handle a rank-1 integer
        // counterpart. Reject mismatches here (rather than panicking deep
        // inside codegen or producing silently wrong code) so users see a
        // clean ProcessError.
        let lhs_ty = &node.inputs[0].ty;
        let rhs_ty = &node.inputs[1].ty;
        if let (ArgType::Shape(_), other) | (other, ArgType::Shape(_)) = (lhs_ty, rhs_ty)
            && other.is_on_device()
        {
            let other_dtype = other.elem_type();
            if !(other_dtype.is_int() || other_dtype.is_uint()) {
                return Err(ProcessError::TypeMismatch {
                    expected: "integer-typed tensor when combined with Shape".to_string(),
                    actual: format!("{other_dtype:?}"),
                });
            }
            if other.rank() != 1 {
                return Err(ProcessError::Custom(format!(
                    "Mod: Shape combined with rank-{} {other_dtype:?} tensor is not supported (expected rank 1)",
                    other.rank()
                )));
            }
        }

        // Output type is same as input with broadcasting
        crate::processor::same_as_input_broadcast(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract fmod attribute
        let fmod = match node.attrs.get("fmod") {
            Some(AttributeValue::Int64(value)) => {
                // TODO: Validate fmod is 0 or 1 - Values other than 0 or 1 are undefined in spec - burn/crates/onnx-ir/src/node/modulo.rs:120
                *value != 0
            }
            _ => false, // Default value as per ONNX spec
        };

        let config = ModConfig::new(fmod);
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Mod(ModNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node() -> crate::ir::RawNode {
        TestNodeBuilder::new(NodeType::Mod, "test_mod")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_f32("result", 2, None)
            .build()
    }

    #[test]
    fn test_mod_config_default() {
        let node = create_test_node();
        let mut node = node;
        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.fmod, false); // Should default to false
    }

    #[test]
    fn test_mod_config_with_fmod_0() {
        let mut node = create_test_node();
        node.attrs
            .insert("fmod".to_string(), AttributeValue::Int64(0));
        let mut node = node;
        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.fmod, false);
    }

    #[test]
    fn test_mod_config_with_fmod_1() {
        let mut node = create_test_node();
        node.attrs
            .insert("fmod".to_string(), AttributeValue::Int64(1));
        let mut node = node;
        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.fmod, true);
    }

    /// Mod on Shape + float tensor must be rejected. The codegen arm
    /// for this combination materializes the Shape as a rank-1 Int
    /// tensor and calls `.remainder()` / `.fmod()` on it, so the
    /// counterpart must be integer-typed.
    #[test]
    fn shape_mod_float_tensor_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::Mod, "mod_shape_float")
            .input_shape("lhs", 3)
            .input_tensor_f32("rhs", 1, None)
            .output_tensor_f32("out", 1, None)
            .build();

        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::TypeMismatch { .. })),
            "expected TypeMismatch, got {result:?}"
        );
    }

    /// Mod on Shape + rank-2 int tensor must be rejected. Codegen
    /// assumes the counterpart is rank 1.
    #[test]
    fn shape_mod_rank2_int_tensor_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::Mod, "mod_shape_rank2")
            .input_shape("lhs", 3)
            .input_tensor_i64("rhs", 2, None)
            .output_tensor_i64("out", 2, None)
            .build();

        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        match result {
            Err(ProcessError::Custom(msg)) => assert!(
                msg.contains("rank-2") && msg.contains("Mod"),
                "expected rank-2 message, got: {msg}"
            ),
            other => panic!("expected Custom ProcessError, got {other:?}"),
        }
    }

    /// Mod on Shape + rank-1 int tensor is the legitimate case and
    /// must be accepted.
    #[test]
    fn shape_mod_rank1_int_tensor_accepted() {
        let mut node = TestNodeBuilder::new(NodeType::Mod, "mod_shape_rank1")
            .input_shape("lhs", 3)
            .input_tensor_i64("rhs", 1, None)
            .output_tensor_i64("out", 1, None)
            .build();

        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_ok(), "expected Ok, got {result:?}");
    }
}
