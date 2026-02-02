//! # GRU
//!
//! Gated Recurrent Unit recurrent neural network.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GRU.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
//! - **Opset 3**: Updated types
//! - **Opset 7**: Added `layout` attribute
//! - **Opset 14**: Added `layout` attribute with batch-first option
//! - **Opset 22**: Added bfloat16 support

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Direction of GRU processing
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GruDirection {
    /// Process sequence from start to end
    #[default]
    Forward,
    /// Process sequence from end to start
    Reverse,
    /// Process in both directions
    Bidirectional,
}

/// Activation function for GRU gates.
///
/// This enum represents all activation functions defined in the ONNX GRU spec.
/// Not all of these are supported by burn-nn; unsupported activations will
/// cause an error during burn-onnx code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GruActivationFunction {
    /// Sigmoid activation (default for gates)
    #[default]
    Sigmoid,
    /// Hyperbolic tangent (default for hidden)
    Tanh,
    /// Rectified Linear Unit
    Relu,
    /// Hard sigmoid approximation: max(0, min(1, alpha*x + beta))
    HardSigmoid,
    /// Leaky ReLU: max(alpha*x, x)
    LeakyRelu,
    /// Thresholded ReLU: x if x > alpha else 0
    ThresholdedRelu,
    /// Scaled Tanh: alpha * tanh(beta * x)
    ScaledTanh,
    /// Exponential Linear Unit: x if x >= 0 else alpha * (exp(x) - 1)
    Elu,
    /// Softsign: x / (1 + |x|)
    Softsign,
    /// Softplus: log(1 + exp(x))
    Softplus,
    /// Affine transformation: alpha * x + beta
    Affine,
}

impl std::str::FromStr for GruActivationFunction {
    type Err = ProcessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sigmoid" => Ok(GruActivationFunction::Sigmoid),
            "tanh" => Ok(GruActivationFunction::Tanh),
            "relu" => Ok(GruActivationFunction::Relu),
            "hardsigmoid" => Ok(GruActivationFunction::HardSigmoid),
            "leakyrelu" => Ok(GruActivationFunction::LeakyRelu),
            "thresholdedrelu" => Ok(GruActivationFunction::ThresholdedRelu),
            "scaledtanh" => Ok(GruActivationFunction::ScaledTanh),
            "elu" => Ok(GruActivationFunction::Elu),
            "softsign" => Ok(GruActivationFunction::Softsign),
            "softplus" => Ok(GruActivationFunction::Softplus),
            "affine" => Ok(GruActivationFunction::Affine),
            _ => Err(ProcessError::InvalidAttribute {
                name: "activations".to_string(),
                reason: format!(
                    "Unknown ONNX activation '{}'. Valid activations: Sigmoid, Tanh, Relu, HardSigmoid, LeakyRelu, ThresholdedRelu, ScaledTanh, Elu, Softsign, Softplus, Affine",
                    s
                ),
            }),
        }
    }
}

impl std::str::FromStr for GruDirection {
    type Err = ProcessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "forward" => Ok(GruDirection::Forward),
            "reverse" => Ok(GruDirection::Reverse),
            "bidirectional" => Ok(GruDirection::Bidirectional),
            _ => Err(ProcessError::InvalidAttribute {
                name: "direction".to_string(),
                reason: format!(
                    "Invalid direction '{}'. Must be 'forward', 'reverse', or 'bidirectional'",
                    s
                ),
            }),
        }
    }
}

impl GruDirection {
    /// Returns the number of directions (1 for forward/reverse, 2 for bidirectional)
    pub fn num_directions(&self) -> usize {
        match self {
            GruDirection::Forward | GruDirection::Reverse => 1,
            GruDirection::Bidirectional => 2,
        }
    }
}

/// Configuration for GRU operations
#[derive(Debug, Clone, new)]
#[allow(clippy::too_many_arguments)]
pub struct GruConfig {
    /// Size of the input features
    pub input_size: usize,
    /// Number of neurons in the hidden layer (required ONNX attribute)
    pub hidden_size: usize,
    /// Direction of GRU processing
    pub direction: GruDirection,
    /// Whether bias is present (input B is provided)
    pub has_bias: bool,
    /// Whether initial hidden state is provided
    pub has_initial_h: bool,
    /// Tensor layout: false = seq_length major (default), true = batch_size major
    pub batch_first: bool,
    /// Cell state clipping threshold (None = no clipping)
    pub clip: Option<f32>,
    /// Whether to apply the linear transformation before multiplying by the reset gate
    pub linear_before_reset: bool,
    /// Activation function for update/reset gates (default: Sigmoid)
    pub gate_activation: GruActivationFunction,
    /// Activation function for hidden gate (default: Tanh)
    pub hidden_activation: GruActivationFunction,
}

/// Node representation for GRU operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GruNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: GruConfig,
}

pub(crate) struct GruProcessor;

impl NodeProcessor for GruProcessor {
    type Config = GruConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            // Inputs: X, W, R (required), B, sequence_lens, initial_h (optional)
            inputs: InputSpec::Range(3, 6),
            // Outputs: Y, Y_h (all optional, but at least one should be present)
            outputs: OutputSpec::Range(0, 2),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // W (weights) and R (recurrence weights) are typically constants
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }
        // B (bias) is optional but typically constant
        if node.inputs.len() > 3 && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }
        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate input tensor (X)
        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // GRU expects 3D input: [seq_length, batch_size, input_size]
        if input_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "GRU expects input tensor of rank 3, got rank {}",
                input_tensor.rank
            )));
        }

        // Validate weight tensor (W)
        let weight_tensor = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        if weight_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "GRU expects weight tensor (W) of rank 3, got rank {}",
                weight_tensor.rank
            )));
        }

        // Validate recurrence weight tensor (R)
        let recurrence_tensor = match &node.inputs[2].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[2].ty),
                });
            }
        };

        if recurrence_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "GRU expects recurrence weight tensor (R) of rank 3, got rank {}",
                recurrence_tensor.rank
            )));
        }

        // Validate optional bias tensor (B) if present and not empty
        if node.inputs.len() > 3
            && !node.inputs[3].is_optional()
            && let ArgType::Tensor(tensor) = &node.inputs[3].ty
            && tensor.rank != 2
        {
            return Err(ProcessError::Custom(format!(
                "GRU expects bias tensor (B) of rank 2, got rank {}",
                tensor.rank
            )));
        }

        // Validate optional initial_h if present and not empty
        if node.inputs.len() > 5
            && !node.inputs[5].is_optional()
            && let ArgType::Tensor(tensor) = &node.inputs[5].ty
            && tensor.rank != 3
        {
            return Err(ProcessError::Custom(format!(
                "GRU expects initial_h tensor of rank 3, got rank {}",
                tensor.rank
            )));
        }

        // Extract config for validation
        let _config = self.extract_config(node, opset)?;

        // Validate sequence_lens is not used
        if node.inputs.len() > 4 && !node.inputs[4].is_optional() {
            return Err(ProcessError::Custom(
                "GRU sequence_lens input is not yet supported. All sequences must have the same length.".to_string(),
            ));
        }

        // Infer output types
        // Output 0: Y - all hidden states [seq_length, num_directions, batch_size, hidden_size]
        if !node.outputs.is_empty() {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: input_tensor.dtype,
                rank: 4,
                static_shape: None,
            });
        }

        // Output 1: Y_h - final hidden state [num_directions, batch_size, hidden_size]
        if node.outputs.len() > 1 {
            node.outputs[1].ty = ArgType::Tensor(TensorType {
                dtype: input_tensor.dtype,
                rank: 3,
                static_shape: None,
            });
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Get input_size from weight tensor or input tensor
        let weight_input = &node.inputs[1];
        let x_input = &node.inputs[0];

        let input_size = if let ArgType::Tensor(t) = &weight_input.ty {
            if let Some(shape) = &t.static_shape {
                if shape.len() == 3 {
                    Some(shape[2])
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let input_size = input_size.or_else(|| {
            weight_input.value().and_then(|data| {
                if data.shape.len() == 3 {
                    Some(data.shape[2])
                } else {
                    None
                }
            })
        });

        let input_size = input_size.or_else(|| {
            if let ArgType::Tensor(t) = &x_input.ty {
                if let Some(shape) = &t.static_shape {
                    if shape.len() >= 3 {
                        Some(shape[2])
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        });

        let input_size = input_size.ok_or_else(|| {
            ProcessError::Custom(
                "GRU: cannot determine input_size - weight tensor (W) and input tensor (X) must have static shape or W must be a constant".to_string()
            )
        })?;

        // Extract hidden_size from attributes (required)
        let hidden_size = node
            .attrs
            .get("hidden_size")
            .ok_or_else(|| ProcessError::MissingAttribute("hidden_size".to_string()))?
            .clone()
            .into_i64() as usize;

        // Extract direction (default: "forward")
        let direction = node
            .attrs
            .get("direction")
            .map(|v| v.clone().into_string())
            .unwrap_or_else(|| "forward".to_string());
        let direction: GruDirection = direction.parse()?;

        // Extract layout (default: 0 = seq_length major)
        let layout = node
            .attrs
            .get("layout")
            .map(|v| v.clone().into_i64())
            .unwrap_or(0);
        let batch_first = layout == 1;

        // Check optional inputs
        let has_bias = node.inputs.len() > 3 && !node.inputs[3].is_optional();
        let has_initial_h = node.inputs.len() > 5 && !node.inputs[5].is_optional();

        // Extract clip threshold (default: None)
        let clip = node.attrs.get("clip").and_then(|v| {
            let val = v.clone().into_f32();
            if val > 0.0 { Some(val) } else { None }
        });

        // Extract linear_before_reset (default: false)
        let linear_before_reset = node
            .attrs
            .get("linear_before_reset")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(false);

        // Extract activations (default: Sigmoid, Tanh for each direction)
        // ONNX format: [f, g] for unidirectional, [f, g, f, g] for bidirectional
        // f = gate activation (z, r gates), g = hidden activation
        let (gate_activation, hidden_activation) = if let Some(activations) =
            node.attrs.get("activations")
        {
            let acts = activations.clone().into_strings();
            if acts.is_empty() {
                (GruActivationFunction::Sigmoid, GruActivationFunction::Tanh)
            } else if acts.len() >= 2 {
                let gate: GruActivationFunction = acts[0].parse()?;
                let hidden: GruActivationFunction = acts[1].parse()?;

                if direction == GruDirection::Bidirectional && acts.len() >= 4 {
                    let gate2: GruActivationFunction = acts[2].parse()?;
                    let hidden2: GruActivationFunction = acts[3].parse()?;

                    if gate != gate2 || hidden != hidden2 {
                        return Err(ProcessError::Custom(
                                "GRU bidirectional with different activations per direction is not supported. Both directions must use the same activations.".to_string(),
                            ));
                    }
                }

                (gate, hidden)
            } else {
                return Err(ProcessError::Custom(format!(
                    "GRU activations must have at least 2 elements, got {}",
                    acts.len()
                )));
            }
        } else {
            (GruActivationFunction::Sigmoid, GruActivationFunction::Tanh)
        };

        Ok(GruConfig::new(
            input_size,
            hidden_size,
            direction,
            has_bias,
            has_initial_h,
            batch_first,
            clip,
            linear_before_reset,
            gate_activation,
            hidden_activation,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Gru(GruNode {
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

    fn create_gru_node(
        hidden_size: i64,
        direction: Option<&str>,
        layout: Option<i64>,
        num_outputs: usize,
    ) -> RawNode {
        let num_directions = match direction {
            Some("bidirectional") => 2,
            _ => 1,
        };

        let mut builder = TestNodeBuilder::new(NodeType::Gru, "test_gru")
            // X: [seq_length=10, batch_size=2, input_size=4]
            .input_tensor_f32("X", 3, Some(vec![10, 2, 4]))
            // W: [num_directions, 3*hidden_size, input_size]
            .input_tensor_f32_data(
                "W",
                vec![0.0; num_directions * 3 * hidden_size as usize * 4],
                vec![num_directions, 3 * hidden_size as usize, 4],
            )
            // R: [num_directions, 3*hidden_size, hidden_size]
            .input_tensor_f32_data(
                "R",
                vec![0.0; num_directions * 3 * hidden_size as usize * hidden_size as usize],
                vec![
                    num_directions,
                    3 * hidden_size as usize,
                    hidden_size as usize,
                ],
            )
            .attr_int("hidden_size", hidden_size);

        if let Some(dir) = direction {
            builder = builder.attr_string("direction", dir);
        }

        if let Some(lay) = layout {
            builder = builder.attr_int("layout", lay);
        }

        // Add outputs
        for i in 0..num_outputs {
            let output_name = match i {
                0 => "Y",
                1 => "Y_h",
                _ => unreachable!(),
            };
            let output_rank = if i == 0 { 4 } else { 3 };
            builder = builder.output_tensor_f32(output_name, output_rank, None);
        }

        builder.build_with_graph_data(14)
    }

    #[test]
    fn test_gru_config_basic() {
        let node = create_gru_node(8, None, None, 2);
        let processor = GruProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert_eq!(config.input_size, 4);
        assert_eq!(config.hidden_size, 8);
        assert_eq!(config.direction, GruDirection::Forward);
        assert!(!config.has_bias);
        assert!(!config.has_initial_h);
        assert!(!config.batch_first);
        assert!(!config.linear_before_reset);
    }

    #[test]
    fn test_gru_config_bidirectional() {
        let node = create_gru_node(8, Some("bidirectional"), None, 2);
        let processor = GruProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert_eq!(config.direction, GruDirection::Bidirectional);
        assert_eq!(config.direction.num_directions(), 2);
    }

    #[test]
    fn test_gru_config_batch_first() {
        let node = create_gru_node(8, None, Some(1), 2);
        let processor = GruProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert!(config.batch_first);
    }

    #[test]
    fn test_gru_type_inference() {
        let mut node = create_gru_node(8, None, None, 2);
        let processor = GruProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 14, &prefs).unwrap();

        // Y: [seq_length, num_directions, batch_size, hidden_size] -> rank 4
        assert!(matches!(&node.outputs[0].ty, ArgType::Tensor(t) if t.rank == 4));
        // Y_h: [num_directions, batch_size, hidden_size] -> rank 3
        assert!(matches!(&node.outputs[1].ty, ArgType::Tensor(t) if t.rank == 3));
    }

    #[test]
    fn test_gru_direction_parsing() {
        assert_eq!(
            "forward".parse::<GruDirection>().unwrap(),
            GruDirection::Forward
        );
        assert_eq!(
            "reverse".parse::<GruDirection>().unwrap(),
            GruDirection::Reverse
        );
        assert_eq!(
            "bidirectional".parse::<GruDirection>().unwrap(),
            GruDirection::Bidirectional
        );
        assert_eq!(
            "FORWARD".parse::<GruDirection>().unwrap(),
            GruDirection::Forward
        );
        assert!("invalid".parse::<GruDirection>().is_err());
    }

    #[test]
    fn test_gru_spec() {
        let processor = GruProcessor;
        let spec = processor.spec();

        assert_eq!(spec.min_opset, 1);
        assert!(spec.max_opset.is_none());
    }
}
