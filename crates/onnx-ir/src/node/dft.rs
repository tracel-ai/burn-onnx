//! # DFT (Discrete Fourier Transform)
//!
//! Computes the discrete Fourier Transform (or its inverse) of the input.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__DFT.html>
//!
//! ## Opset Versions
//! - **Opset 17**: Initial version
//! - **Opset 20**: Updated type constraints
//!
//! ## Supported Configurations
//!
//! Only forward real-input DFT maps to Burn's signal API:
//! - Forward real DFT with `onesided=1`: maps to `burn::tensor::signal::rfft`
//! - Forward real DFT with `onesided=0`: rfft + conjugate symmetry reconstruction
//!
//! Not supported (will produce a codegen error):
//! - Inverse DFT (`inverse=1`): ONNX uses complex-to-complex IDFT, but Burn only
//!   provides `irfft` (inverse of real FFT), which is a different operation
//! - Complex-to-complex forward DFT (complex input with trailing dim = 2)
//!
//! ## Type Constraints
//! - **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
//! - **T2**: tensor(int32), tensor(int64)

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorData, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, validate_opset,
};

/// Extract a scalar integer from tensor data, supporting both i32 and i64 dtypes.
fn extract_scalar_int(data: TensorData, name: &str) -> Result<i64, ProcessError> {
    if let Ok(slice) = data.as_slice::<i64>() {
        slice.first().copied().ok_or_else(|| {
            ProcessError::Custom(format!(
                "DFT: {name} constant must contain at least one element"
            ))
        })
    } else if let Ok(slice) = data.as_slice::<i32>() {
        slice.first().copied().map(i64::from).ok_or_else(|| {
            ProcessError::Custom(format!(
                "DFT: {name} constant must contain at least one element"
            ))
        })
    } else {
        Err(ProcessError::Custom(format!(
            "DFT: {name} constant must have type int32 or int64"
        )))
    }
}

/// Configuration for the DFT operation
#[derive(Debug, Clone, Default)]
pub struct DftConfig {
    /// Whether to compute the inverse DFT (default: false)
    pub inverse: bool,
    /// Whether to produce onesided output for real input (default: false)
    pub onesided: bool,
    /// The axis along which to perform the DFT (resolved to positive index)
    pub axis: usize,
    /// Optional DFT length (None means use the signal dimension size)
    pub dft_length: Option<usize>,
    /// Whether the input is real (trailing dim = 1) or complex (trailing dim = 2)
    pub is_real_input: bool,
}

/// Node representation for DFT operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct DftNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: DftConfig,
}

pub(crate) struct DftProcessor;

impl NodeProcessor for DftProcessor {
    type Config = DftConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 17,
            max_opset: None,
            inputs: InputSpec::Range(1, 3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, opset: usize) -> Result<(), ProcessError> {
        // Lift dft_length (input[1]) if present and constant
        if let Some(input) = node.inputs.get(1)
            && !input.is_optional()
            && input.is_constant()
        {
            node.inputs[1].to_static()?;
        }

        // Lift axis (input[2]) if present and constant (opset 20+ only; opset 17-19 uses attribute)
        if opset >= 20
            && let Some(input) = node.inputs.get(2)
            && !input.is_optional()
            && input.is_constant()
        {
            node.inputs[2].to_static()?;
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

        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        if input_tensor.rank < 2 {
            return Err(ProcessError::Custom(format!(
                "DFT: input must have rank >= 2 (got rank {}). \
                 The last dimension represents real/complex components.",
                input_tensor.rank
            )));
        }

        // Determine if input is real or complex from the trailing dimension.
        // The trailing dim must be statically known (1 = real, 2 = complex) because
        // the codegen strategy is fundamentally different for each case.
        let is_real_input = match &input_tensor.static_shape {
            Some(shape) => match shape.last() {
                Some(Some(1)) => true,
                Some(Some(2)) => false,
                Some(Some(d)) => {
                    return Err(ProcessError::Custom(format!(
                        "DFT: last dimension must be 1 (real) or 2 (complex), got {d}"
                    )));
                }
                _ => {
                    return Err(ProcessError::Custom(
                        "DFT: last dimension must be statically known as 1 (real) or 2 (complex). \
                         Ensure the ONNX model has static shapes on the DFT input."
                            .to_string(),
                    ));
                }
            },
            None => {
                return Err(ProcessError::Custom(
                    "DFT: input shape must be statically known. \
                     The last dimension determines real (1) vs complex (2) input."
                        .to_string(),
                ));
            }
        };

        // Extract attributes
        let inverse = node
            .attrs
            .get("inverse")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(false);

        let onesided = node
            .attrs
            .get("onesided")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(false);

        // Reject unsupported configurations early with clear errors
        if inverse {
            return Err(ProcessError::Custom(
                "DFT: inverse DFT (inverse=1) is not supported. \
                 Burn's irfft is the inverse of rfft (onesided real FFT), \
                 which differs from ONNX's complex-to-complex inverse DFT. \
                 A full ifft implementation in Burn is needed to support this."
                    .to_string(),
            ));
        }

        if !is_real_input {
            return Err(ProcessError::Custom(
                "DFT: complex-to-complex DFT is not supported by Burn's current signal API. \
                 Only real-input forward DFT (onesided or full) is supported."
                    .to_string(),
            ));
        }

        // Validate: complex input cannot be onesided (unreachable now, but kept for spec completeness)
        if !is_real_input && onesided {
            return Err(ProcessError::Custom(
                "DFT: onesided output is not possible with complex input".to_string(),
            ));
        }

        // Resolve axis (default is -2, i.e. last signal dimension)
        let axis = self.resolve_axis(node, &input_tensor, opset)?;

        // Try to extract dft_length for shape inference (if constant)
        let static_dft_length = match node.inputs.get(1) {
            Some(input) if !input.is_optional() => match input.value() {
                Some(data) => Some(extract_scalar_int(data, "dft_length")? as usize),
                None => None,
            },
            _ => None,
        };

        // Output is always complex: same shape but last dim = 2
        // The signal axis uses dft_length if provided, otherwise the input dimension.
        // If onesided, the signal axis dimension becomes floor(N/2) + 1.
        let out_rank = input_tensor.rank;
        let out_static_shape = if let Some(shape) = &input_tensor.static_shape {
            let mut out_shape = shape.clone();
            // Last dim is always 2 (complex output)
            *out_shape.last_mut().unwrap() = Some(2);

            // Effective DFT length: dft_length if provided, otherwise the input dim
            let effective_n = static_dft_length.or_else(|| out_shape.get(axis).copied().flatten());

            if let Some(n) = effective_n {
                out_shape[axis] = Some(if onesided { n / 2 + 1 } else { n });
            }

            Some(out_shape)
        } else {
            None
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: input_tensor.dtype,
            rank: out_rank,
            static_shape: out_static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, opset: usize) -> Result<Self::Config, ProcessError> {
        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        let inverse = node
            .attrs
            .get("inverse")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(false);

        let onesided = node
            .attrs
            .get("onesided")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(false);

        let axis = self.resolve_axis(node, &input_tensor, opset)?;

        // Extract dft_length from input[1] if present
        let dft_length = match node.inputs.get(1) {
            Some(input) if !input.is_optional() => match input.value() {
                Some(data) => {
                    let val = extract_scalar_int(data, "dft_length")?;
                    if val <= 0 {
                        return Err(ProcessError::Custom(
                            "DFT: dft_length must be a positive integer".to_string(),
                        ));
                    }
                    Some(val as usize)
                }
                None => {
                    return Err(ProcessError::Custom(
                        "DFT: dft_length must be a compile-time constant".to_string(),
                    ));
                }
            },
            _ => None,
        };

        // Determine is_real_input from trailing dim (validated in infer_types)
        let is_real_input = match &input_tensor.static_shape {
            Some(shape) => matches!(shape.last(), Some(Some(1))),
            _ => true, // infer_types rejects unknown shapes, so this is a safe fallback
        };

        Ok(DftConfig {
            inverse,
            onesided,
            axis,
            dft_length,
            is_real_input,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Dft(DftNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

impl DftProcessor {
    /// Resolve the axis parameter.
    ///
    /// In opset 17-19, axis is an attribute. In opset 20+, it moved to input[2].
    fn resolve_axis(
        &self,
        node: &RawNode,
        input_tensor: &TensorType,
        opset: usize,
    ) -> Result<usize, ProcessError> {
        let rank = input_tensor.rank as i64;

        let raw_axis: i64 = if opset < 20 {
            // Opset 17-19: axis is an attribute
            node.attrs
                .get("axis")
                .map(|v| v.clone().into_i64())
                .unwrap_or(-2)
        } else {
            // Opset 20+: axis is an optional input
            match node.inputs.get(2) {
                Some(input) if !input.is_optional() => match input.value() {
                    Some(data) => extract_scalar_int(data, "axis")?,
                    None => {
                        return Err(ProcessError::Custom(
                            "DFT: axis must be a compile-time constant".to_string(),
                        ));
                    }
                },
                _ => -2,
            }
        };

        // Validate axis range: [-r, -2] union [0, r-2]
        // The last dimension is reserved for real/complex encoding, so axis = -1 is invalid.
        let valid =
            (raw_axis >= -rank && raw_axis <= -2) || (raw_axis >= 0 && raw_axis <= rank - 2);
        if !valid {
            return Err(ProcessError::Custom(format!(
                "DFT: axis {raw_axis} out of valid range [-{rank}, -2] or [0, {}] for input rank {rank}",
                rank - 2
            )));
        }

        // Resolve negative axis
        let axis = if raw_axis < 0 {
            (rank + raw_axis) as usize
        } else {
            raw_axis as usize
        };

        Ok(axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;

    #[test]
    fn test_dft_forward_real_onesided() {
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, Some(vec![1, 16, 1]))
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::I64,
                    rank: 0,
                    static_shape: None,
                }),
            ) // optional dft_length
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::I64,
                    rank: 0,
                    static_shape: None,
                }),
            ) // optional axis
            .output_tensor_f32("output", 0, None)
            .attr_int("onesided", 1)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.dtype, DType::F32);
                // onesided: signal dim 16 -> 16/2+1 = 9, last dim = 2
                let shape = t.static_shape.as_ref().unwrap();
                assert_eq!(shape, &vec![Some(1), Some(9), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_dft_forward_real_twosided() {
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, Some(vec![1, 16, 1]))
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                // Non-onesided: signal dim stays 16, last dim = 2
                let shape = t.static_shape.as_ref().unwrap();
                assert_eq!(shape, &vec![Some(1), Some(16), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_dft_inverse_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, Some(vec![1, 16, 2]))
            .output_tensor_f32("output", 0, None)
            .attr_int("inverse", 1)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("inverse"));
    }

    #[test]
    fn test_dft_complex_input_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, Some(vec![1, 16, 2]))
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("complex-to-complex")
        );
    }

    #[test]
    fn test_dft_unknown_shape_rejected() {
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, None)
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("statically known"));
    }

    #[test]
    fn test_dft_rank_too_low() {
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 1, Some(vec![8]))
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_dft_opset_too_low() {
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, Some(vec![1, 16, 1]))
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_dft_preserves_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f64("input", 3, Some(vec![1, 16, 1]))
            .output_tensor_f32("output", 0, None)
            .attr_int("onesided", 1)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.dtype, DType::F64),
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_dft_config_extraction() {
        let node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, Some(vec![1, 16, 1]))
            .output_tensor_f32("output", 0, None)
            .attr_int("inverse", 1)
            .attr_int("onesided", 1)
            .build();

        let processor = DftProcessor;
        let config = processor.extract_config(&node, 17).unwrap();

        assert!(config.inverse);
        assert!(config.onesided);
        assert_eq!(config.axis, 1); // default -2 on rank 3 = 1
        assert_eq!(config.dft_length, None);
        assert!(config.is_real_input);
    }

    #[test]
    fn test_dft_axis_out_of_range() {
        // Opset 17: axis is an attribute
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, Some(vec![1, 16, 1]))
            .output_tensor_f32("output", 0, None)
            .attr_int("axis", 2) // axis=2 is last dim (invalid for rank 3)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("axis"));
    }

    #[test]
    fn test_dft_axis_out_of_range_opset20() {
        // Opset 20: axis is an input
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 3, Some(vec![1, 16, 1]))
            .input_tensor_i64_data("dft_length", vec![16], vec![])
            .input_tensor_i64_data("axis", vec![2], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(20);

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 20, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("axis"));
    }

    #[test]
    fn test_dft_with_axis_attribute_opset17() {
        // Opset 17: axis is an attribute
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 4, Some(vec![2, 8, 16, 1]))
            .output_tensor_f32("output", 0, None)
            .attr_int("onesided", 1)
            .attr_int("axis", 1)
            .build();

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                let shape = t.static_shape.as_ref().unwrap();
                assert_eq!(shape, &vec![Some(2), Some(5), Some(16), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_dft_with_axis_input_opset20() {
        // Opset 20: axis is an input
        let mut node = TestNodeBuilder::new(NodeType::Dft, "test_dft")
            .input_tensor_f32("input", 4, Some(vec![2, 8, 16, 1]))
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::I64,
                    rank: 0,
                    static_shape: None,
                }),
            ) // optional dft_length
            .input_tensor_i64_data("axis", vec![1], vec![])
            .output_tensor_f32("output", 0, None)
            .attr_int("onesided", 1)
            .build_with_graph_data(20);

        let processor = DftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 20, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                // axis=1, dim=8, onesided -> 8/2+1=5
                let shape = t.static_shape.as_ref().unwrap();
                assert_eq!(shape, &vec![Some(2), Some(5), Some(16), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }
}
