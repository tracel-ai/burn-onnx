use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use strum_macros::EnumString;


#[derive(Debug, Clone, PartialEq, Default, EnumString)]
pub enum RnnDirection {
    #[default]
    Forward,
    Reverse,
    Bidirectional,
}

impl RnnDirection {
    pub fn num_directions(&self) -> usize {
        match self {
            RnnDirection::Forward | RnnDirection::Reverse => 1,
            RnnDirection::Bidirectional => 2,
        }
    }
}


#[derive(Debug, Clone, PartialEq, Copy, Default, Eq, EnumString)]
pub enum RnnActivationFunction{
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


#[derive(Debug, Clone, new)]
pub struct RnnConfig{
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
    pub batch_first: bool,
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

    fn lift_constants(&self, node: &RawNode, _opset: usize) -> Result<(), ProcessError>{
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
        node: &RawNode,
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
        
        // RNN expects 3D weights matrix: [num_directions, 4*hidden_size, input_size]
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

        Ok(())
    }
}