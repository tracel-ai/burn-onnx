use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum RnnDirection {
    #[default]
    Forward,
    Reverse,
    Bidirectional,
}

impl std::str::FromStr for RnnDirection {
    type Err = ProcessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "forward" => Ok(RnnDirection::Forward),
            "reverse" => Ok(RnnDirection::Reverse),
            "bidirectional" => Ok(RnnDirection::Bidirectional),
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

impl RnnDirection {
    pub fn num_directions(&self) -> usize {
        match self {
            RnnDirection::Forward | RnnDirection::Reverse => 1,
            RnnDirection::Bidirectional => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy, Default, Eq)]
pub enum RnnActivationFunction {
    #[default]
    Tanh,
    Relu,
    Sigmoid,
    Affine,
    LeakyRelu,
    ThresholdedRelu,
    ScaledTanh,
    HardSigmoid,
    Elu,
    Softsign,
    Softplus,
}

impl std::str::FromStr for RnnActivationFunction {
    type Err = ProcessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // ONNX activation names (case-insensitive matching)
        match s.to_lowercase().as_str() {
            "sigmoid" => Ok(RnnActivationFunction::Sigmoid),
            "tanh" => Ok(RnnActivationFunction::Tanh),
            "relu" => Ok(RnnActivationFunction::Relu),
            "hardsigmoid" => Ok(RnnActivationFunction::HardSigmoid),
            "leakyrelu" => Ok(RnnActivationFunction::LeakyRelu),
            "thresholdedrelu" => Ok(RnnActivationFunction::ThresholdedRelu),
            "scaledtanh" => Ok(RnnActivationFunction::ScaledTanh),
            "elu" => Ok(RnnActivationFunction::Elu),
            "softsign" => Ok(RnnActivationFunction::Softsign),
            "softplus" => Ok(RnnActivationFunction::Softplus),
            "affine" => Ok(RnnActivationFunction::Affine),
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

#[derive(Debug, Clone, new)]
pub struct RnnConfig {
    // Size of the input features
    pub input_size: usize,
    // Number of neurons in the hidden layer (required ONNX attribute)
    pub hidden_size: usize,
    // Direction of RNN processing
    pub direction: RnnDirection,
    // Whether bias is present (input B is provided)
    pub has_bias: bool,
    // Whether initial hidden state is provided
    pub has_initial_h: bool,
    /// Tensor layout: false = seq_length major (default), true = batch_size major
    pub batch_first: bool,
    /// Hidden state clipping threshold (None = no clipping)
    pub clip: Option<f32>,
    /// Activation function for hidden state output (default: Tanh)
    pub hidden_activation: RnnActivationFunction,
}

#[derive(Debug, Clone, NodeBuilder)]
pub struct RnnNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: RnnConfig,
}

pub(crate) struct RnnProcessor;

impl NodeProcessor for RnnProcessor {
    type Config = RnnConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            // Inputs:
            //     Required: X, W, R
            //     Optional: B, sequence_lens, initial_h
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

        // RNN expects 3D input: [seq_length, batch_size, input_size] or [batch_size, seq_length, input_size]
        if input_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "RNN expects input tensor of rank 3, got rank {}",
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

        // RNN expects 3D weights matrix: [num_directions, hidden_size, input_size]
        if weight_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "RNN expects weight tensor (W) of rank 3, got rank {}",
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

        // RNN expects 3D weights matrix: [num_directions, hidden_size, hidden_size]
        if recurrence_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "RNN expects recurrence weight tensor (R) of rank 3, got rank {}",
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
                "RNN expects bias tensor (B) of rank 2, got rank {}",
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
                "RNN expects initial_h tensor of rank 3, got rank {}",
                tensor.rank
            )));
        }

        // Extract config for validation for sequence_lens checks below
        let _config = self.extract_config(node, opset)?;

        // Infer output types based on which outputs are requested
        // Output 0: Y - all hidden states [seq_length, num_directions, batch_size, hidden_size]
        if !node.outputs.is_empty() {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: input_tensor.dtype,
                rank: 4, // [seq_length, num_directions, batch_size, hidden_size]
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

        // Validate sequence_lens is not used (not supported in Burn)
        if node.inputs.len() > 4 && !node.inputs[4].is_optional() {
            return Err(ProcessError::Custom(
                "RNN sequence_lens input is not yet supported. All sequences must have the same length.".to_string(),
            ));
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Get input_size - can be derived from:
        // 1. Weight tensor W shape: [num_directions, hidden_size, input_size] -> input_size is W[2]
        // 2. Input tensor X shape: [seq_length, batch_size, input_size] -> input_size is X[2]
        //
        // We try multiple sources since weight tensors may be dynamically computed (e.g., after
        // Concat/Slice operations) while the input tensor often has static shape from the model input.
        let weight_input = &node.inputs[1];
        let x_input = &node.inputs[0];
        log::debug!(
            "RNN extract_config: X input type={:?}, W input type={:?}",
            x_input.ty,
            weight_input.ty
        );

        // Try to get input_size from weight tensor's static_shape
        let input_size = if let ArgType::Tensor(t) = &weight_input.ty {
            if let Some(shape) = &t.static_shape {
                if shape.len() == 3 {
                    log::debug!("RNN: using input_size from W static_shape: {:?}", shape[2]);
                    shape[2]
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Extract clip threshold (default: None)
        let clip = node.attrs.get("clip").and_then(|v| {
            let val = v.clone().into_f32();
            if val > 0.0 { Some(val) } else { None }
        });

        // Fallback: try to get input_size from weight constant data
        let input_size = input_size.or_else(|| {
            weight_input.value().and_then(|data| {
                if data.shape.len() == 3 {
                    log::debug!(
                        "RNN: using input_size from W constant value: {}",
                        data.shape[2]
                    );
                    Some(data.shape[2])
                } else {
                    None
                }
            })
        });

        // Fallback: try to get input_size from input tensor X's static_shape
        // X has shape [seq_length, batch_size, input_size] so input_size is X[2]
        let input_size = input_size.or_else(|| {
            if let ArgType::Tensor(t) = &x_input.ty {
                if let Some(shape) = &t.static_shape {
                    if shape.len() >= 3 {
                        log::debug!("RNN: using input_size from X static_shape: {:?}", shape[2]);
                        shape[2]
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
                "RNN: cannot determine input_size - weight tensor (W) and input tensor (X) must have static shape or W must be a constant".to_string()
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
        let direction: RnnDirection = direction.parse()?;

        // Extract layout (default: 0 = seq_length major)
        // layout attribute was added in opset 14, but we support it for all versions
        let layout = node
            .attrs
            .get("layout")
            .map(|v| v.clone().into_i64())
            .unwrap_or(0);
        let batch_first = layout == 1;

        // Check optional inputs
        let has_bias = node.inputs.len() > 3 && !node.inputs[3].is_optional();
        let has_initial_h = node.inputs.len() > 5 && !node.inputs[5].is_optional();

        // Extract activations (default: Tanh for each direction)
        // f = hidden activation
        let hidden_activation = if let Some(activations) = node.attrs.get("activations") {
            let acts = activations.clone().into_strings();
            if acts.is_empty() {
                // Empty means use defaults
                RnnActivationFunction::Tanh
            } else if acts.len() >= 1 {
                let hidden: RnnActivationFunction = acts[0].parse()?;

                // For bidirectional, verify both directions use the same activations
                if direction == RnnDirection::Bidirectional && acts.len() >= 2 {
                    let hidden2: RnnActivationFunction = acts[1].parse()?;
                    if hidden != hidden2 {
                        return Err(ProcessError::Custom(
                                "RNN bidirectional with different activations per direction is not supported. Both directions must use the same activations.".to_string(),
                            ));
                    }
                }

                hidden
            } else {
                return Err(ProcessError::Custom(format!(
                    "RNN activations must have at least 3 elements, got {}",
                    acts.len()
                )));
            }
        } else {
            // No activations attribute means use defaults
            RnnActivationFunction::Tanh
        };

        let config = RnnConfig::new(
            input_size,
            hidden_size,
            direction,
            has_bias,
            has_initial_h,
            batch_first,
            clip,
            hidden_activation,
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Rnn(RnnNode {
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

    fn create_rnn_node(
        hidden_size: i64,
        direction: Option<&str>,
        layout: Option<i64>,
        num_outputs: usize,
    ) -> RawNode {
        let num_directions = match direction {
            Some("bidirectional") => 2,
            _ => 1,
        };

        let mut builder = TestNodeBuilder::new(NodeType::Rnn, "test_rnn")
            // X: [seq_length=10, batch_size=2, input_size=4]
            .input_tensor_f32("X", 3, Some(vec![10, 2, 4]))
            // W: [num_directions, hidden_size, input_size]
            .input_tensor_f32_data(
                "W",
                vec![0.0; num_directions * hidden_size as usize * 4],
                vec![num_directions, hidden_size as usize, 4],
            )
            // R: [num_directions, hidden_size, hidden_size]
            .input_tensor_f32_data(
                "R",
                vec![0.0; num_directions * hidden_size as usize * hidden_size as usize],
                vec![num_directions, hidden_size as usize, hidden_size as usize],
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

        builder.build_with_graph_data(14) // opset 14
    }

    #[test]
    fn test_rnn_config_basic() {
        let node = create_rnn_node(8, None, None, 2);
        let processor = RnnProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert_eq!(config.input_size, 4);
        assert_eq!(config.hidden_size, 8);
        assert_eq!(config.direction, RnnDirection::Forward);
        assert!(!config.has_bias);
        assert!(!config.has_initial_h);
        assert!(!config.batch_first);
    }

    #[test]
    fn test_rnn_config_bidirectional() {
        let node = create_rnn_node(8, Some("bidirectional"), None, 2);
        let processor = RnnProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert_eq!(config.direction, RnnDirection::Bidirectional);
        assert_eq!(config.direction.num_directions(), 2);
    }

    #[test]
    fn test_rnn_config_batch_first() {
        let node = create_rnn_node(8, None, Some(1), 2);
        let processor = RnnProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert!(config.batch_first);
    }

    #[test]
    fn test_rnn_type_inference() {
        let mut node = create_rnn_node(8, None, None, 2);
        let processor = RnnProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 14, &prefs).unwrap();

        // Y: [seq_length, num_directions, batch_size, hidden_size] -> rank 4
        assert!(matches!(&node.outputs[0].ty, ArgType::Tensor(t) if t.rank == 4));
        // Y_h: [num_directions, batch_size, hidden_size] -> rank 3
        assert!(matches!(&node.outputs[1].ty, ArgType::Tensor(t) if t.rank == 3));
    }

    #[test]
    fn test_rnn_direction_parsing() {
        assert_eq!(
            "forward".parse::<RnnDirection>().unwrap(),
            RnnDirection::Forward
        );
        assert_eq!(
            "reverse".parse::<RnnDirection>().unwrap(),
            RnnDirection::Reverse
        );
        assert_eq!(
            "bidirectional".parse::<RnnDirection>().unwrap(),
            RnnDirection::Bidirectional
        );
        assert_eq!(
            "FORWARD".parse::<RnnDirection>().unwrap(),
            RnnDirection::Forward
        );
        assert!("invalid".parse::<RnnDirection>().is_err());
    }

    #[test]
    fn test_rnn_spec() {
        let processor = RnnProcessor;
        let spec = processor.spec();

        assert_eq!(spec.min_opset, 1);
        assert!(spec.max_opset.is_none());
    }
}
