//! # GELU (Gaussian Error Linear Unit) Operation
//!
//! Element-wise GELU activation operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Gelu.html>
//!
//! ## Attributes
//! - `approximate` (string, default `"none"`): `"none"` for the exact erf form, `"tanh"` for
//!   the tanh approximation
//!
//! ## Type Constraints
//!
//! T: Float tensor types
//!
//! ## Opset Versions
//! - **Opset 20+**: Initial version

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};
use onnx_ir_derive::NodeBuilder;

/// How GELU is evaluated, from the ONNX `approximate` attribute.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GeluApproximate {
    /// The exact form, `x * Φ(x)` with the Gaussian CDF expressed through erf.
    #[default]
    Exact,
    /// The tanh approximation of the Gaussian CDF.
    Tanh,
}

/// Configuration for the Gelu operation.
#[derive(Debug, Clone, Default)]
pub struct GeluConfig {
    /// Which form of GELU to evaluate.
    pub approximate: GeluApproximate,
}

/// Node representation for Gelu operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GeluNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: GeluConfig,
}

/// Node processor for GELU operation
pub(crate) struct GeluProcessor;

impl NodeProcessor for GeluProcessor {
    type Config = GeluConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 20,
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
        validate_opset(opset, 20)?;
        self.extract_config(node, opset)?;
        same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let approximate = match node.attrs.get("approximate") {
            None => GeluApproximate::Exact,
            Some(value) => match value.clone().into_string().as_str() {
                "none" => GeluApproximate::Exact,
                "tanh" => GeluApproximate::Tanh,
                other => {
                    return Err(ProcessError::InvalidAttribute {
                        name: "approximate".to_string(),
                        reason: format!("expected \"none\" or \"tanh\", got \"{other}\""),
                    });
                }
            },
        };
        Ok(GeluConfig { approximate })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Gelu(GeluNode {
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

    fn create_test_node(approximate: Option<&str>) -> RawNode {
        let builder = TestNodeBuilder::new(NodeType::Gelu, "test_gelu")
            .input_tensor_f32("X", 2, None)
            .output_tensor_f32("Y", 2, None);
        match approximate {
            Some(value) => builder.attr_string("approximate", value),
            None => builder,
        }
        .build()
    }

    #[test]
    fn test_gelu_config_default_is_exact() {
        let node = create_test_node(None);
        let config = GeluProcessor.extract_config(&node, 20).unwrap();
        assert_eq!(config.approximate, GeluApproximate::Exact);
    }

    #[test]
    fn test_gelu_config_tanh() {
        let node = create_test_node(Some("tanh"));
        let config = GeluProcessor.extract_config(&node, 20).unwrap();
        assert_eq!(config.approximate, GeluApproximate::Tanh);
    }

    #[test]
    fn test_gelu_rejects_unknown_approximation() {
        let mut node = create_test_node(Some("sigmoid"));
        let result = GeluProcessor.infer_types(&mut node, 20, &OutputPreferences::new());
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }
}
