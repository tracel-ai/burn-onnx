//! # LRN (Local Response Normalization)
//!
//! Local Response Normalization as proposed in the AlexNet paper. Normalizes
//! over local input regions across the channel dimension.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Lrn.html>
//!
//! ## Formula
//! `Y[n,c,...] = X[n,c,...] / (bias + alpha/size * sum(X[n,i,...]^2))^beta`
//! where the sum is over `i` in the local channel window of size `size`.
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
//! - **Opset 13**: Added bfloat16 type support

use crate::ArgType;
use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use burn_tensor::DType;
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for LRN (Local Response Normalization) operations.
#[derive(Debug, Clone, new)]
pub struct LrnConfig {
    /// Scaling parameter
    pub alpha: f32,
    /// The exponent
    pub beta: f32,
    /// Bias constant
    pub bias: f32,
    /// Number of channels to sum over
    pub size: i64,
}

/// Node representation for LRN operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct LrnNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LrnConfig,
}

pub(crate) struct LrnProcessor;

impl NodeProcessor for LrnProcessor {
    type Config = LrnConfig;

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
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate node attributes
        let _ = self.extract_config(node, opset)?;

        // Validate input tensor types
        let arg = node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("Missing input".to_string()))?;
        let ArgType::Tensor(ref tensor_ty) = arg.ty else {
            return Err(ProcessError::TypeMismatch {
                expected: "Input should be a tensor".to_string(),
                actual: format!("{:?}", arg.ty),
            });
        };
        if tensor_ty.rank < 4 {
            return Err(ProcessError::TypeMismatch {
                expected: "Expecting a tensor of at least rank 4".to_string(),
                actual: format!("Got a rank-{:?} tensor instead", tensor_ty.rank),
            });
        }
        if opset >= 13
            && !matches!(
                tensor_ty.dtype,
                DType::BF16 | DType::F16 | DType::F32 | DType::F64
            )
        {
            return Err(ProcessError::TypeMismatch {
                expected: "Only BF16, F16, F32, F64 tensor dtypes are supported".to_string(),
                actual: format!("{:?}", tensor_ty.dtype),
            });
        } else if !matches!(tensor_ty.dtype, DType::F16 | DType::F32 | DType::F64) {
            return Err(ProcessError::TypeMismatch {
                expected: "Only F16, F32, F64 tensor dtypes are supported".to_string(),
                actual: format!("{:?}", tensor_ty.dtype),
            });
        }

        crate::processor::same_as_input_broadcast(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract `alpha`, `beta` and `bias` attributes
        const ALPHA_DEFAULT: f32 = 0.0001;
        const BETA_DEFAULT: f32 = 0.75;
        const BIAS_DEFAULT: f32 = 1.0;
        let alpha = node
            .attrs
            .get("alpha")
            .map(|val| val.clone().into_f32())
            .unwrap_or(ALPHA_DEFAULT);
        let beta = node
            .attrs
            .get("beta")
            .map(|val| val.clone().into_f32())
            .unwrap_or(BETA_DEFAULT);
        let bias = node
            .attrs
            .get("bias")
            .map(|val| val.clone().into_f32())
            .unwrap_or(BIAS_DEFAULT);

        // Validate that `size` exists, and is of type `int`
        let size = node
            .attrs
            .get("size")
            .map(|val| val.clone().into_i64())
            .ok_or_else(|| ProcessError::MissingAttribute("size".to_string()))?;

        Ok(LrnConfig {
            alpha,
            beta,
            bias,
            size,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Configi extraction failed");

        Node::Lrn(LrnNode {
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

    fn create_test_node(alpha: f32, beta: f32, bias: f32, size: i64) -> RawNode {
        TestNodeBuilder::new(NodeType::Lrn, "test_lrn")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .attr_float("beta", beta)
            .attr_float("bias", bias)
            .attr_int("size", size)
            .build()
    }

    #[test]
    fn test_lrn_config_defaults() {
        // Build a node with only `size` set — alpha/beta/bias should fall back to ONNX defaults.
        let mut node = create_test_node(0.0, 0.0, 0.0, 5);
        node.attrs.retain(|k, _| k == "size"); // strip alpha/beta/bias

        let config = LrnProcessor.extract_config(&node, 13).unwrap();

        assert!((config.alpha - 0.0001).abs() < 1e-7);
        assert!((config.beta - 0.75).abs() < 1e-7);
        assert!((config.bias - 1.0).abs() < 1e-7);
        assert_eq!(config.size, 5);
    }

    #[test]
    fn test_lrn_config_custom_values() {
        let node = create_test_node(0.001, 0.5, 2.0, 3);

        let config = LrnProcessor.extract_config(&node, 13).unwrap();

        assert!((config.alpha - 0.001).abs() < 1e-6);
        assert!((config.beta - 0.5).abs() < 1e-7);
        assert!((config.bias - 2.0).abs() < 1e-7);
        assert_eq!(config.size, 3);
    }

    #[test]
    fn test_lrn_missing_size_attr_errors() {
        let mut node = create_test_node(0.0001, 0.75, 1.0, 5);
        node.attrs.retain(|k, _| k != "size");

        let result = LrnProcessor.extract_config(&node, 13);

        assert!(matches!(
            result,
            Err(ProcessError::MissingAttribute(ref s)) if s == "size"
        ))
    }

    #[test]
    fn test_lrn_infer_types_preserves_shape() {
        let mut node = TestNodeBuilder::new(NodeType::Lrn, "test_lrn")
            .input_tensor_f32("X", 4, Some(vec![1, 5, 3, 3]))
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", 0.0001)
            .attr_float("beta", 0.75)
            .attr_float("bias", 1.0)
            .attr_int("size", 5)
            .build();

        let prefs = OutputPreferences::new();
        LrnProcessor.infer_types(&mut node, 13, &prefs).unwrap();

        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(t) => {
                assert_eq!(
                    t.static_shape,
                    Some(vec![Some(1), Some(5), Some(3), Some(3)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_lrn_infer_types_preserves_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::Lrn, "test_lrn")
            .input_tensor_f64("X", 4, None)
            .output_tensor_f64("Y", 4, None)
            .attr_float("alpha", 0.0001)
            .attr_float("beta", 0.75)
            .attr_float("bias", 1.0)
            .attr_int("size", 5)
            .build();

        let prefs = OutputPreferences::new();
        LrnProcessor.infer_types(&mut node, 13, &prefs).unwrap();

        match &node.outputs[0].ty {
            crate::ir::ArgType::Tensor(t) => {
                assert_eq!(t.dtype, crate::ir::DType::F64);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
