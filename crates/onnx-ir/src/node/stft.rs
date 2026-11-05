//! # STFT (Short-Time Fourier Transform)
//!
//! Computes the Short-time Fourier Transform of the signal.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__STFT.html>
//!
//! ## Opset Versions
//! - **Opset 17**: Initial version.
//!
//! ## Supported Configurations
//!
//! Only real-input STFT maps to Burn's `stft` API:
//! - Real input: rank-3 `[batch, signal_length, 1]` (ONNX spec form) or rank-2
//!   `[batch, signal_length]` (produced by PyTorch's ONNX exporter and accepted
//!   by ONNX Runtime), `onesided = 0 | 1`.
//! - `frame_step` and `frame_length` may be compile-time constants or runtime scalars
//!   (e.g. graph inputs). Runtime values are read on the host when the model runs; the output
//!   shape then has unknown frame/frequency dimensions.
//! - Optional window tensor, rank 1, length equal to `frame_length`.
//!
//! Any `frame_length` maps to Burn's `burn::tensor::signal::stft`, which uses a radix-2 FFT
//! for power-of-two lengths and Bluestein's algorithm otherwise.
//!
//! Not supported (rejected with a clear error):
//! - Complex-to-complex STFT (signal rank 3, trailing dim = 2): Burn's STFT
//!   has no complex input path.
//!
//! ## `frame_length` default
//!
//! The ONNX spec's default for `frame_length` when absent is `signal_length`.
//! We instead fall back to the `window` tensor's static shape when
//! `frame_length` is absent. Real ONNX models that omit `frame_length`
//! virtually always provide a window, so this deviation is safe in practice.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, RuntimeInputRef, TensorData, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, validate_opset,
};

const OP_NAME: &str = "STFT";

/// Extract a scalar integer from tensor data, supporting both i32 and i64 dtypes.
fn extract_scalar_int(data: TensorData, name: &str) -> Result<i64, ProcessError> {
    if let Ok(slice) = data.as_slice::<i64>() {
        slice.first().copied().ok_or_else(|| {
            ProcessError::Custom(format!(
                "{OP_NAME}: {name} constant must contain at least one element"
            ))
        })
    } else if let Ok(slice) = data.as_slice::<i32>() {
        slice.first().copied().map(i64::from).ok_or_else(|| {
            ProcessError::Custom(format!(
                "{OP_NAME}: {name} constant must contain at least one element"
            ))
        })
    } else {
        Err(ProcessError::Custom(format!(
            "{OP_NAME}: {name} constant must have type int32 or int64"
        )))
    }
}

/// A scalar STFT input (`frame_step` or `frame_length`).
#[derive(Debug, Clone)]
pub enum StftInput {
    /// Value known at compile time.
    Static(usize),
    /// Value read at run time from a scalar int32/int64 input.
    Runtime(RuntimeInputRef),
}

impl Default for StftInput {
    fn default() -> Self {
        Self::Static(0)
    }
}

impl StftInput {
    /// The compile-time value, if known.
    pub fn as_static(&self) -> Option<usize> {
        match self {
            Self::Static(v) => Some(*v),
            Self::Runtime(_) => None,
        }
    }
}

/// Configuration for the STFT operation.
#[derive(Debug, Clone, Default)]
pub struct StftConfig {
    /// If true, produces onesided output (`n_fft / 2 + 1` frequency bins).
    pub onesided: bool,
    /// Number of samples to step between successive DFTs (Burn's `hop_length`).
    pub frame_step: StftInput,
    /// Size of each DFT window (Burn's `n_fft`).
    pub frame_length: StftInput,
    /// Whether a window tensor is provided as input.
    pub has_window: bool,
}

/// Node representation for the STFT operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct StftNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: StftConfig,
}

pub(crate) struct StftProcessor;

impl StftProcessor {
    /// Read a scalar `frame_step` / `frame_length` input: its positive value when it is a
    /// constant, or a runtime reference when it is a scalar int32/int64 computed at run time.
    fn extract_scalar_input(
        node: &RawNode,
        index: usize,
        name: &str,
    ) -> Result<StftInput, ProcessError> {
        let input = &node.inputs[index];
        let Some(data) = input.value() else {
            let is_int_scalar = match &input.ty {
                ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => {
                    matches!(dtype, DType::I32 | DType::I64)
                }
                _ => false,
            };
            if !is_int_scalar {
                return Err(ProcessError::Custom(format!(
                    "{OP_NAME}: runtime {name} must be an int32 or int64 scalar, got {:?}",
                    input.ty
                )));
            }
            return Ok(StftInput::Runtime(RuntimeInputRef::new(
                input.name.clone(),
                index,
            )));
        };
        let val = extract_scalar_int(data, name)?;
        if val <= 0 {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: {name} must be a positive integer, got {val}"
            )));
        }
        let val = usize::try_from(val).map_err(|_| {
            ProcessError::Custom(format!("{OP_NAME}: {name} must fit in usize, got {val}"))
        })?;
        Ok(StftInput::Static(val))
    }

    /// Extract `frame_step` from input[1] (required).
    fn extract_frame_step(node: &RawNode) -> Result<StftInput, ProcessError> {
        if node.inputs.len() < 2 {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: missing required frame_step input"
            )));
        }
        Self::extract_scalar_input(node, 1, "frame_step")
    }

    /// Extract `frame_length` from input[3] if present, otherwise infer from `window` shape.
    ///
    /// Returns `None` only if neither is provided (which the caller rejects).
    fn extract_frame_length(node: &RawNode) -> Result<Option<StftInput>, ProcessError> {
        if let Some(input) = node.inputs.get(3)
            && !input.is_optional()
        {
            return Self::extract_scalar_input(node, 3, "frame_length").map(Some);
        }

        // Fall back to inferring from the window's static shape.
        if let Some(window) = node.inputs.get(2)
            && !window.is_optional()
            && let ArgType::Tensor(t) = &window.ty
            && let Some(shape) = &t.static_shape
            && let Some(Some(w)) = shape.first()
        {
            return Ok(Some(StftInput::Static(*w)));
        }

        Ok(None)
    }

    /// Compute the output shape `[batch, n_frames, n_freqs, 2]`. Dimensions that depend on a
    /// runtime `frame_step` / `frame_length` (`None`) or on an unknown signal length are `None`.
    fn compute_output_shape(
        signal_shape: &[Option<usize>],
        frame_step: Option<usize>,
        frame_length: Option<usize>,
        onesided: bool,
    ) -> Vec<Option<usize>> {
        let batch = signal_shape.first().copied().flatten();
        let signal_len = signal_shape.get(1).copied().flatten();

        let n_frames = match (signal_len, frame_step, frame_length) {
            (Some(l), Some(step), Some(len)) => {
                Some(if l < len { 0 } else { 1 + (l - len) / step })
            }
            _ => None,
        };

        let n_freqs = frame_length.map(|len| if onesided { len / 2 + 1 } else { len });

        vec![batch, n_frames, n_freqs, Some(2)]
    }
}

impl NodeProcessor for StftProcessor {
    type Config = StftConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 17,
            max_opset: None,
            inputs: InputSpec::Range(2, 4),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        if let Some(input) = node.inputs.get(1)
            && !input.is_optional()
            && input.is_constant()
        {
            node.inputs[1].to_static()?;
        }

        if let Some(input) = node.inputs.get(3)
            && !input.is_optional()
            && input.is_constant()
        {
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
        validate_opset(opset, 17)?;

        let signal_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.clone(),
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };

        // Accept rank 3 (ONNX spec: [batch, signal_length, 1|2]) and rank 2
        // ([batch, signal_length]). PyTorch's ONNX exporter emits the rank-2
        // form; ORT accepts it as real-valued.
        match signal_tensor.rank {
            2 => { /* implicit real input, no trailing dim to inspect */ }
            3 => {
                let is_real_input = match &signal_tensor.static_shape {
                    Some(shape) => match shape.last() {
                        Some(Some(1)) => true,
                        Some(Some(2)) => false,
                        Some(Some(d)) => {
                            return Err(ProcessError::Custom(format!(
                                "{OP_NAME}: signal last dimension must be 1 (real) or 2 (complex), got {d}"
                            )));
                        }
                        _ => {
                            return Err(ProcessError::Custom(format!(
                                "{OP_NAME}: signal last dimension must be statically known as 1 or 2"
                            )));
                        }
                    },
                    None => {
                        return Err(ProcessError::Custom(format!(
                            "{OP_NAME}: signal shape must be statically known \
                             (last dim determines real/complex)"
                        )));
                    }
                };

                if !is_real_input {
                    return Err(ProcessError::Custom(format!(
                        "{OP_NAME}: complex-to-complex STFT is not supported. \
                         Burn's stft requires real-valued input."
                    )));
                }
            }
            r => {
                return Err(ProcessError::Custom(format!(
                    "{OP_NAME}: signal input must have rank 2 [batch, signal_length] \
                     or rank 3 [batch, signal_length, 1|2], got rank {r}"
                )));
            }
        }

        // Optional window (input[2]) must be rank 1 if present.
        if let Some(window) = node.inputs.get(2)
            && !window.is_optional()
        {
            match &window.ty {
                ArgType::Tensor(t) => {
                    if t.rank != 1 {
                        return Err(ProcessError::Custom(format!(
                            "{OP_NAME}: window must have rank 1, got rank {}",
                            t.rank
                        )));
                    }
                }
                other => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor for window".to_string(),
                        actual: format!("{other:?}"),
                    });
                }
            }
        }

        let frame_step = Self::extract_frame_step(node)?;
        let frame_length = Self::extract_frame_length(node)?.ok_or_else(|| {
            ProcessError::Custom(format!(
                "{OP_NAME}: frame_length must be provided as an input \
                 or inferable from a window with a static shape"
            ))
        })?;
        let static_step = frame_step.as_static();
        let static_length = frame_length.as_static();

        // Burn's stft handles any n_fft (Bluestein for non-power-of-two lengths). Only reject
        // `0`, which is nonsensical. A runtime frame_length is checked by burn when it runs.
        if static_length == Some(0) {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: frame_length must be >= 1, got 0"
            )));
        }

        if let (Some(step), Some(length)) = (static_step, static_length)
            && step > length
        {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: frame_step ({step}) must be <= frame_length ({length}) \
                 to satisfy the overlap-add constraint"
            )));
        }

        // Cross-check window length against frame_length when both are statically known.
        if let Some(length) = static_length
            && let Some(window) = node.inputs.get(2)
            && !window.is_optional()
            && let ArgType::Tensor(t) = &window.ty
            && let Some(shape) = &t.static_shape
            && let Some(Some(w)) = shape.first()
            && *w != length
        {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: window length ({w}) must equal frame_length ({length})"
            )));
        }

        let onesided = node
            .attrs
            .get("onesided")
            .map(|v| v.clone().into_i64().map(|x| x != 0))
            .transpose()?
            .unwrap_or(true);

        let static_shape = signal_tensor
            .static_shape
            .as_ref()
            .map(|s| Self::compute_output_shape(s, static_step, static_length, onesided));

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: signal_tensor.dtype,
            rank: 4,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let frame_step = Self::extract_frame_step(node)?;
        let frame_length = Self::extract_frame_length(node)?.ok_or_else(|| {
            ProcessError::Custom(format!("{OP_NAME}: frame_length could not be resolved"))
        })?;
        let onesided = node
            .attrs
            .get("onesided")
            .map(|v| v.clone().into_i64().map(|x| x != 0))
            .transpose()?
            .unwrap_or(true);
        let has_window = matches!(
            node.inputs.get(2),
            Some(input) if !input.is_optional()
        );

        Ok(StftConfig {
            onesided,
            frame_step,
            frame_length,
            has_window,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        // Constant frame_step (input[1]) and frame_length (input[3]) were lifted to Static by
        // lift_constants, so they are dropped from the generated forward() signature. Runtime
        // ones stay as forward() inputs.
        Ok(Node::Stft(StftNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;

    fn builder_with_signal(signal_shape: Vec<usize>) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Stft, "test_stft").input_tensor_f32(
            "signal",
            signal_shape.len(),
            Some(signal_shape),
        )
    }

    #[test]
    fn test_stft_real_onesided_default() {
        let mut node = builder_with_signal(vec![1, 64, 1])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                let shape = t.static_shape.as_ref().unwrap();
                // n_frames = 1 + (64 - 16) / 4 = 13, n_freqs onesided = 16/2+1 = 9
                assert_eq!(shape, &vec![Some(1), Some(13), Some(9), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_stft_real_full() {
        let mut node = builder_with_signal(vec![2, 32, 1])
            .input_tensor_i64_data("frame_step", vec![8], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![8], vec![])
            .output_tensor_f32("output", 0, None)
            .attr_int("onesided", 0)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                let shape = t.static_shape.as_ref().unwrap();
                // n_frames = 1 + (32-8)/8 = 4, n_freqs full = 8
                assert_eq!(shape, &vec![Some(2), Some(4), Some(8), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_stft_with_window() {
        let mut node = builder_with_signal(vec![1, 128, 1])
            .input_tensor_i64_data("frame_step", vec![16], vec![])
            .input_tensor_f32("window", 1, Some(vec![32]))
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::I64,
                    rank: 0,
                    static_shape: None,
                }),
            )
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        let config = processor.extract_config(&node, 17).unwrap();
        assert_eq!(config.frame_step.as_static(), Some(16));
        assert_eq!(config.frame_length.as_static(), Some(32));
        assert!(config.has_window);
    }

    #[test]
    fn test_stft_real_rank2_input() {
        // PyTorch's ONNX exporter emits real STFT signal as rank 2
        // [batch, signal_length] with no trailing 1. ORT accepts it and
        // so should we.
        let mut node = builder_with_signal(vec![1, 64])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                let shape = t.static_shape.as_ref().unwrap();
                // Same as the rank-3 real case: batch=1, n_frames=13, n_freqs=9
                assert_eq!(shape, &vec![Some(1), Some(13), Some(9), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_stft_rank1_rejected() {
        let mut node = builder_with_signal(vec![64])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 17, &prefs).unwrap_err();
        assert!(err.to_string().contains("rank 1"));
    }

    #[test]
    fn test_stft_complex_rejected() {
        let mut node = builder_with_signal(vec![1, 64, 2])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("complex"));
    }

    #[test]
    fn test_stft_non_power_of_two_accepted() {
        // Non-pow2 frame_length (20, matching Kokoro). Burn's stft handles
        // it through Bluestein's algorithm.
        let mut node = builder_with_signal(vec![1, 128, 1])
            .input_tensor_i64_data("frame_step", vec![5], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![20], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                let shape = t.static_shape.as_ref().unwrap();
                // n_frames = 1 + (128 - 20) / 5 = 22, n_freqs onesided = 11
                assert_eq!(shape, &vec![Some(1), Some(22), Some(11), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }

        let config = processor.extract_config(&node, 17).unwrap();
        assert_eq!(config.frame_length.as_static(), Some(20));
        assert_eq!(config.frame_step.as_static(), Some(5));
    }

    #[test]
    fn test_stft_runtime_frame_step_with_window() {
        // As in the ONNX backend test `test_stft_with_window`: frame_step is a graph input,
        // frame_length comes from the window's static shape.
        let mut node = builder_with_signal(vec![1, 128, 1])
            .input_scalar_i64("frame_step")
            .input_tensor_f32("window", 1, Some(vec![16]))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        processor
            .infer_types(&mut node, 17, &OutputPreferences::new())
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                // n_frames depends on the runtime frame_step; n_freqs = 16/2+1 = 9
                assert_eq!(t.static_shape, Some(vec![Some(1), None, Some(9), Some(2)]));
            }
            _ => panic!("Expected Tensor output"),
        }

        let config = processor.extract_config(&node, 17).unwrap();
        match &config.frame_step {
            StftInput::Runtime(r) => {
                assert_eq!(r.name, "frame_step");
                assert_eq!(r.input_index, 1);
            }
            other => panic!("expected runtime frame_step, got {other:?}"),
        }
        assert_eq!(config.frame_length.as_static(), Some(16));
    }

    #[test]
    fn test_stft_runtime_frame_step_and_length() {
        // As in the ONNX backend test `test_stft`: both are graph inputs.
        let mut node = builder_with_signal(vec![1, 128, 1])
            .input_scalar_i64("frame_step")
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_scalar_i64("frame_length")
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        processor
            .infer_types(&mut node, 17, &OutputPreferences::new())
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.static_shape, Some(vec![Some(1), None, None, Some(2)]));
            }
            _ => panic!("Expected Tensor output"),
        }

        let config = processor.extract_config(&node, 17).unwrap();
        assert!(matches!(config.frame_step, StftInput::Runtime(_)));
        match &config.frame_length {
            StftInput::Runtime(r) => {
                assert_eq!(r.name, "frame_length");
                assert_eq!(r.input_index, 3);
            }
            other => panic!("expected runtime frame_length, got {other:?}"),
        }
    }

    #[test]
    fn test_stft_runtime_frame_step_must_be_int_scalar() {
        let mut node = builder_with_signal(vec![1, 64, 1])
            .input_tensor_i64("frame_step", 1, None) // runtime, but a rank-1 tensor
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("runtime frame_step must be an int32 or int64 scalar")
        );
    }

    #[test]
    fn test_stft_frame_step_gt_frame_length_rejected() {
        let mut node = builder_with_signal(vec![1, 64, 1])
            .input_tensor_i64_data("frame_step", vec![32], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("overlap-add"));
    }

    #[test]
    fn test_stft_window_mismatch_rejected() {
        let mut node = builder_with_signal(vec![1, 128, 1])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .input_tensor_f32("window", 1, Some(vec![20])) // mismatch
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("window length"));
    }

    #[test]
    fn test_stft_infer_frame_length_from_window() {
        let node = builder_with_signal(vec![1, 128, 1])
            .input_tensor_i64_data("frame_step", vec![8], vec![])
            .input_tensor_f32("window", 1, Some(vec![16]))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let config = processor.extract_config(&node, 17).unwrap();
        assert_eq!(config.frame_length.as_static(), Some(16));
        assert!(config.has_window);
    }

    #[test]
    fn test_stft_opset_too_low() {
        let mut node = builder_with_signal(vec![1, 64, 1])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(16);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("opset"));
    }

    // compute_output_shape boundary tests: exercise the three n_frames branches
    // (signal < frame_length, signal == frame_length, non-exact division) that
    // the higher-level tests all skip because their lengths divide evenly.

    #[test]
    fn test_stft_shape_signal_shorter_than_frame_length() {
        // signal_length (8) < frame_length (16) -> n_frames = 0
        let shape = StftProcessor::compute_output_shape(
            &[Some(1), Some(8), Some(1)],
            Some(4),
            Some(16),
            true,
        );
        assert_eq!(shape, vec![Some(1), Some(0), Some(9), Some(2)]);
    }

    #[test]
    fn test_stft_shape_signal_equals_frame_length() {
        // signal_length == frame_length -> exactly 1 frame
        let shape = StftProcessor::compute_output_shape(
            &[Some(1), Some(16), Some(1)],
            Some(8),
            Some(16),
            true,
        );
        assert_eq!(shape, vec![Some(1), Some(1), Some(9), Some(2)]);
    }

    #[test]
    fn test_stft_shape_non_exact_division() {
        // signal_length=33, frame_length=16, frame_step=8:
        // (33 - 16) / 8 = 17/8 = 2 (floor) -> n_frames = 3
        let shape = StftProcessor::compute_output_shape(
            &[Some(1), Some(33), Some(1)],
            Some(8),
            Some(16),
            true,
        );
        assert_eq!(shape, vec![Some(1), Some(3), Some(9), Some(2)]);
    }
}
