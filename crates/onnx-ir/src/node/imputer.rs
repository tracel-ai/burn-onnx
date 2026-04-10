//! # Imputer
//!
//! Replaces missing values in the input tensor with specified replacement values.
//! The Imputer operator supports replacing NaN values and specified values with imputation values.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx_aionnxml_Imputer.html>
//!
//! ## Type Constraints
//!
//! - T: tensor(float), tensor(double), tensor(int32), tensor(int64)
//!
//! ## Opset Versions
//!
//! - **Opset 1**: Initial version with basic imputation functionality

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, AttributeValue, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};

/// Configuration for Imputer operation
#[derive(Debug, Clone, Default, new)]
pub struct ImputerConfig {
    /// Imputed values for float data
    pub imputed_value_floats: Option<Vec<f32>>,
    /// Imputed values for integer data
    pub imputed_value_int64s: Option<Vec<i64>>,
    /// Value to replace for float inputs (NaN if not specified)
    pub replaced_value_float: Option<f32>,
    /// Value to replace for integer inputs
    pub replaced_value_int64: Option<i64>,
}

/// Node representation for Imputer operation
#[derive(Debug, Clone, new, NodeBuilder)]
pub struct ImputerNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ImputerConfig,
}

pub(crate) struct ImputerProcessor;

impl NodeProcessor for ImputerProcessor {
    type Config = ImputerConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Output has the same type as input
        same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut imputed_value_floats: Option<Vec<f32>> = None;
        let mut imputed_value_int64s: Option<Vec<i64>> = None;
        let mut replaced_value_float: Option<f32> = None;
        let mut replaced_value_int64: Option<i64> = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "imputed_value_floats" => {
                    if let AttributeValue::Float32s(floats) = value {
                        imputed_value_floats = Some(floats.clone());
                    }
                }
                "imputed_value_int64s" => {
                    if let AttributeValue::Int64s(ints) = value {
                        imputed_value_int64s = Some(ints.clone());
                    }
                }
                "replaced_value_float" => {
                    replaced_value_float = Some(value.clone().into_f32());
                }
                "replaced_value_int64" => {
                    replaced_value_int64 = Some(value.clone().into_i64());
                }
                _ => {}
            }
        }

        Ok(ImputerConfig::new(
            imputed_value_floats,
            imputed_value_int64s,
            replaced_value_float,
            replaced_value_int64,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");
        Node::Imputer(ImputerNode::new(
            builder.name,
            builder.inputs,
            builder.outputs,
            config,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imputer_config_extraction() {
        let config = ImputerConfig::new(Some(vec![0.0]), None, None, None);
        assert!(config.imputed_value_floats.is_some());
        assert_eq!(config.imputed_value_floats.unwrap(), vec![0.0]);
    }

    #[test]
    fn test_imputer_node_builder() {
        let config = ImputerConfig::new(Some(vec![1.0]), None, Some(-999.0), None);
        let node = ImputerNode::new("test_imputer".to_string(), vec![], vec![], config);

        assert_eq!(node.name, "test_imputer");
        assert_eq!(node.inputs.len(), 0);
        assert_eq!(node.outputs.len(), 0);
        assert!(node.config.imputed_value_floats.is_some());
        assert_eq!(node.config.replaced_value_float, Some(-999.0));
    }
}
