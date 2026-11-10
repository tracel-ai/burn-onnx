//! # Random Tensor Generation (RandomNormal, RandomUniform)
//!
//! This module implements processors for ONNX random tensor generation operations that create
//! tensors with random values sampled from specified distributions.
//!
//! **ONNX Specs**:
//! - RandomNormal: <https://onnx.ai/onnx/operators/onnx__RandomNormal.html>
//! - RandomUniform: <https://onnx.ai/onnx/operators/onnx__RandomUniform.html>
//!
//! ## Supported Operations
//!
//! - **RandomNormal**: Generates a tensor with values drawn from a normal distribution
//! - **RandomUniform**: Generates a tensor with values drawn from a uniform distribution
//!
//! Both operations require a `shape` attribute and support an optional `dtype` attribute
//! (defaults to FLOAT if not specified).
//!
//! ## Opset Versions
//!
//! ### RandomNormal
//! - **Opset 1**: Initial version with shape, dtype, mean, scale, and seed attributes.
//!
//! ### RandomUniform
//! - **Opset 1**: Initial version with shape, dtype, high, low, and seed attributes.
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

/// Configuration for RandomNormal operation.
#[derive(Debug, Clone)]
pub struct RandomNormalConfig {
    /// Optional ONNX random seed, preserved without integer truncation.
    pub seed: Option<f32>,
    pub mean: f64,
    pub scale: f64,
    pub shape: Vec<usize>,
}

/// Configuration for RandomUniform operation.
#[derive(Debug, Clone)]
pub struct RandomUniformConfig {
    /// Optional ONNX random seed, preserved without integer truncation.
    pub seed: Option<f32>,
    pub low: f64,
    pub high: f64,
    pub shape: Vec<usize>,
}

/// Enum config that can hold either RandomNormal or RandomUniform config
#[derive(Debug, Clone)]
pub enum RandomConfig {
    Normal(RandomNormalConfig),
    Uniform(RandomUniformConfig),
}

/// Node representation for RandomNormal operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct RandomNormalNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: RandomNormalConfig,
}

/// Node representation for RandomUniform operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct RandomUniformNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: RandomUniformConfig,
}

pub(crate) struct RandomProcessor;

impl NodeProcessor for RandomProcessor {
    type Config = RandomConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(0),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate that this node has zero inputs (Random operations don't take inputs)
        // Random operators should have no inputs, but this isn't validated.
        // Should add: validate_input_count(node, 0) or validate_max_inputs(node, 0)
        // Location: After validate_output_count

        // TODO: RandomNormal mean and scale attributes not validated or extracted
        // ONNX spec defines:
        // - mean (float, default=0.0): Mean of the normal distribution
        // - scale (float, default=1.0): Standard deviation of the normal distribution
        // - seed (float, optional): Random seed for reproducibility
        // These attributes exist in spec but are completely ignored by implementation.
        // Should extract into config and validate ranges (scale > 0).
        // Location: extract_config method

        // TODO: RandomUniform high/low attributes not validated or extracted
        // ONNX spec defines:
        // - high (float, default=1.0): Upper boundary of uniform distribution (exclusive)
        // - low (float, default=0.0): Lower boundary of uniform distribution (inclusive)
        // - seed (float, optional): Random seed for reproducibility
        // These attributes are ignored. Should extract into config and validate low < high.
        // Location: extract_config method

        // TODO: Missing test coverage for non-default distribution parameters
        // Tests only validate basic shape output, not distribution parameters (mean, scale, high, low).
        // Add tests: random_normal_custom_mean_scale, random_uniform_custom_range
        // Note: Testing actual distribution properties is hard, but should at least extract attrs.

        let dtype = node
            .attrs
            .get("dtype")
            .map(|val| {
                let code = val.clone().into_i32()?;
                DataType::from_i32(code).ok_or_else(|| {
                    ProcessError::Custom(format!("unknown dtype attribute value {code}"))
                })
            })
            .transpose()?
            .unwrap_or(DataType::FLOAT);

        let shape = node
            .attrs
            .get("shape")
            .ok_or_else(|| ProcessError::Custom("required shape attribute missing".to_string()))?
            .clone()
            .into_i64s()?;

        let elem_type = match dtype {
            DataType::FLOAT => DType::F32,
            DataType::DOUBLE => DType::F64,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "dtype".to_string(),
                    reason: format!("tensor with type {dtype:?} not supported for random output"),
                });
            }
        };

        let rank = shape.len();

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: elem_type,
            rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let shape = node
            .attrs
            .get("shape")
            .ok_or_else(|| ProcessError::Custom("required shape attribute missing".to_string()))?
            .clone()
            .into_i64s()?;
        let shape: Vec<usize> = shape.into_iter().map(|i| i as usize).collect();

        let seed = node
            .attrs
            .get("seed")
            .map(|v| v.clone().into_f32())
            .transpose()?;
        let config = match node.node_type {
            crate::ir::NodeType::RandomNormal => {
                let mean = node
                    .attrs
                    .get("mean")
                    .map(|v| v.clone().into_f32().map(|x| x as f64))
                    .transpose()?
                    .unwrap_or(0.0);
                let scale = node
                    .attrs
                    .get("scale")
                    .map(|v| v.clone().into_f32().map(|x| x as f64))
                    .transpose()?
                    .unwrap_or(1.0);
                RandomConfig::Normal(RandomNormalConfig {
                    mean,
                    scale,
                    shape,
                    seed,
                })
            }
            crate::ir::NodeType::RandomUniform => {
                let low = node
                    .attrs
                    .get("low")
                    .map(|v| v.clone().into_f32().map(|x| x as f64))
                    .transpose()?
                    .unwrap_or(0.0);
                let high = node
                    .attrs
                    .get("high")
                    .map(|v| v.clone().into_f32().map(|x| x as f64))
                    .transpose()?
                    .unwrap_or(1.0);
                RandomConfig::Uniform(RandomUniformConfig {
                    low,
                    high,
                    shape,
                    seed,
                })
            }
            _ => {
                return Err(ProcessError::Custom(format!(
                    "RandomProcessor does not support node type {:?}",
                    node.node_type
                )));
            }
        };

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(match config {
            RandomConfig::Normal(normal_config) => Node::RandomNormal(RandomNormalNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config: normal_config,
            }),
            RandomConfig::Uniform(uniform_config) => Node::RandomUniform(RandomUniformNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config: uniform_config,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: i32, shape: Vec<i64>) -> RawNode {
        TestNodeBuilder::new(NodeType::RandomNormal, "test_random")
            .output_tensor_f32("output", 0, None) // Rank 0 will be updated
            .attr_int("dtype", dtype as i64)
            .attr_ints("shape", shape)
            .build()
    }

    #[test]
    fn preserves_optional_fractional_seed() {
        let node = TestNodeBuilder::new(NodeType::RandomNormal, "seeded")
            .attr_ints("shape", vec![2, 3])
            .attr_float("seed", 42.5)
            .build();
        let RandomConfig::Normal(config) = RandomProcessor.extract_config(&node, 22).unwrap()
        else {
            panic!("wrong config")
        };
        assert_eq!(config.seed, Some(42.5));
        let mut absent = node;
        absent.attrs.remove("seed");
        let RandomConfig::Normal(config) = RandomProcessor.extract_config(&absent, 22).unwrap()
        else {
            panic!("wrong config")
        };
        assert_eq!(config.seed, None);
        let node = TestNodeBuilder::new(NodeType::RandomUniform, "seeded")
            .attr_ints("shape", vec![2, 3])
            .attr_float("seed", 42.5)
            .build();
        let RandomConfig::Uniform(config) = RandomProcessor.extract_config(&node, 22).unwrap()
        else {
            panic!("wrong config")
        };
        assert_eq!(config.seed, Some(42.5));
        let mut absent = node;
        absent.attrs.remove("seed");
        let RandomConfig::Uniform(config) = RandomProcessor.extract_config(&absent, 22).unwrap()
        else {
            panic!("wrong config")
        };
        assert_eq!(config.seed, None);
    }

    #[test]
    fn test_random_normal_float() {
        let mut node = create_test_node(DataType::FLOAT.value(), vec![2, 3, 4]);
        let processor = RandomProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_normal_double() {
        let mut node = create_test_node(DataType::DOUBLE.value(), vec![5]);
        let processor = RandomProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_normal_missing_shape() {
        // Create node and then manually remove the shape attribute
        let mut node = create_test_node(DataType::FLOAT.value(), vec![2, 3]);
        node.attrs.remove("shape");
        let processor = RandomProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_random_normal_unsupported_type() {
        let mut node = create_test_node(DataType::INT32.value(), vec![2, 3]);
        let processor = RandomProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }
}
