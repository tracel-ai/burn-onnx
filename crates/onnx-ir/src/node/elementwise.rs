//! # Element-wise Operations
//!
//! Processors for element-wise unary and binary operations that operate on tensors element-by-element.
//!
//! **ONNX Specs**: Multiple operations with varying opset requirements
//!
//! ## Opset Versions
//!
//! **Opset 1 Operations:**
//! - **Not**: Logical NOT
//!
//! **Opset 6 Operations:**
//! - **Abs**: Absolute value |x| (improved shape inference)
//! - **Ceil**: Round up to nearest integer (improved shape inference)
//! - **Floor**: Round down to nearest integer (improved shape inference)
//! - **Exp**: Exponential e^x (improved shape inference)
//! - **Log**: Natural logarithm ln(x) (improved shape inference)
//! - **Neg**: Negation -x (improved shape inference)
//! - **Reciprocal**: Reciprocal 1/x (improved shape inference)
//! - **Sqrt**: Square root √x (improved shape inference)
//!
//! **Opset 7 Operations (Trigonometric):**
//! - **Cos**: Cosine
//! - **Sin**: Sine
//! - **Tan**: Tangent
//!
//! **Opset 9 Operations:**
//! - **Erf**: Error function
//! - **Sign**: Sign function (-1, 0, or 1)
//! - **Sinh**: Hyperbolic sine
//! - **Cosh**: Hyperbolic cosine
//! - **Tanh**: Hyperbolic tangent
//!
//! **Opset 11 Operations:**
//! - **Round**: Round to nearest integer (supports optional mode attribute)
//!
//! **Other Operations:**
//! - **Sigmoid**: Sigmoid function 1/(1+e^-x) (Opset 6+)
//! - **Gelu**: Gaussian Error Linear Unit (Opset 20+)
// TODO: Gelu supports 'approximate' attribute (default="none", also "tanh") per ONNX spec - Not extracted or validated - Should add config extraction
//! - **Max**: Element-wise maximum (Opset 1+)
//! - **Min**: Element-wise minimum (Opset 1+)
//! - **BitwiseNot**: Bitwise NOT (Opset 18+)
//!
//!
//! ## Implementation Notes
//! - No opset validation currently performed for binary operations (see TODO at line 108)

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};

/// Node representation for element-wise binary operations
#[derive(Debug, Clone)]
pub struct ElementwiseBinaryNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub node_type: crate::ir::NodeType,
}

/// Node representation for element-wise unary operations
#[derive(Debug, Clone)]
pub struct ElementwiseUnaryNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub node_type: crate::ir::NodeType,
}

/// Node processor for element-wise unary operations that don't yet have
/// dedicated processors (Elu, Selu,
/// ThresholdedRelu). Will be removed as these ops get their own processors.
#[allow(dead_code)]
pub(crate) struct ElementwiseUnaryProcessor;

impl NodeProcessor for ElementwiseUnaryProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        // Determine opset based on operation type
        let min_opset = 1;

        NodeSpec {
            min_opset,
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
        // Validate opset based on operation type
        // Note: Only operations still using ElementwiseUnaryNode are handled here
        let min_opset = match node.node_type {
            // Other activation functions
            crate::ir::NodeType::Elu => 6,
            crate::ir::NodeType::Selu => 6,
            crate::ir::NodeType::ThresholdedRelu => 10,
            _ => {
                return Err(ProcessError::Custom(format!(
                    "Unexpected node type for ElementwiseUnaryProcessor: {:?}",
                    node.node_type
                )));
            }
        };

        crate::processor::validate_opset(opset, min_opset)?;

        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        use crate::ir::NodeType;

        let node = ElementwiseUnaryNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            node_type: builder.node_type.clone(),
        };

        match builder.node_type {
            // Activation functions (still using ElementwiseUnaryNode)
            NodeType::Elu => Node::Elu(node),
            NodeType::Selu => Node::Selu(node),
            NodeType::ThresholdedRelu => Node::ThresholdedRelu(node),
            _ => panic!(
                "Unsupported node type for ElementwiseUnaryProcessor: {:?}",
                builder.node_type
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, NodeType, TensorType};

    #[test]
    fn test_elementwise_unary_processor() {
        let processor = ElementwiseUnaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::RawNode {
            node_type: NodeType::Elu,
            name: "test_elu".to_string(),
            inputs: vec![Argument {
                name: "a".to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 3,
                    static_shape: None,
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "b".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should match input
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.dtype, DType::F32);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unary_unsupported_opset() {
        let processor = ElementwiseUnaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::RawNode {
            node_type: NodeType::ThresholdedRelu,
            name: "test_thresholded_relu".to_string(),
            inputs: vec![Argument {
                name: "a".to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 2,
                    static_shape: None,
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "b".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
        };

        let result = processor.infer_types(&mut node, 9, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::UnsupportedOpset {
                required: 10,
                actual: 9
            })
        ));
    }
}
