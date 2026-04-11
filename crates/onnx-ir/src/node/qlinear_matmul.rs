//! # QLinearMatMul
//!
//! Quantized matrix multiplication: dequantizes both inputs to float, performs matmul,
//! then requantizes the result.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html>
//!
//! ## Opset Versions
//! - **Opset 10**: Initial version with int8/uint8 support.
//! - **Opset 21**: Added float8 type variants.

use burn_tensor::DType;
use burn_tensor::quantization::QuantValue;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::{ArgType, TensorType};

/// Node representation for QLinearMatMul.
///
/// Inputs (in order): a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
#[derive(Debug, Clone, NodeBuilder)]
pub struct QLinearMatMulNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct QLinearMatMulProcessor;

impl NodeProcessor for QLinearMatMulProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Exact(8),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validations performed:
        // 1. Zero-point inputs (a_zero_point, b_zero_point, y_zero_point) must be quantized integer types (I8, U8, F8E5M2, F8E4M3)
        // 2. Operand inputs (a, b) must be tensors, not scalars
        // 3. Each tensor and its zero-point must be of the same dtype
        // 4. Scale inputs (a_scale, b_scale, y_scale) must be floating-point types (BF16, F16, F32)
        // 5. Scale and zero-point pairs must have matching ranks
        // 6. Scale/zero-point ranks must not exceed their corresponding tensor's rank

        let a = &node.inputs[0];
        let a_scale = &node.inputs[1];
        let a_zero_point = &node.inputs[2];
        let b = &node.inputs[3];
        let b_scale = &node.inputs[4];
        let b_zero_point = &node.inputs[5];
        let y_scale = &node.inputs[6];
        let y_zero_point = &node.inputs[7];

        // Validate dtypes for zero points inputs
        for (input, name) in [
            (a_zero_point, "a_zero_point"),
            (b_zero_point, "b_zero_point"),
            (y_zero_point, "y_zero_point"),
        ] {
            let dtype = input.ty.elem_type();
            match dtype {
                DType::I8 | DType::U8 => {}
                // FIXME: Support unsized F8 types (F85ME2UZ and F8M5E2UZ) as required by the spec
                DType::QFloat(quant_scheme)
                    if opset >= 21
                        && matches!(quant_scheme.value, QuantValue::E5M2 | QuantValue::E4M3) => {}
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Only I8, U8, F8M5E2, F8M4E3 tensor dtypes are supported"
                            .to_string(),
                        actual: format!("{name}: {dtype:?}"),
                    });
                }
            }
        }

        // Validate that inputs `a` and `b` are tensors
        for (input, name) in [(a, "a"), (b, "b")] {
            if !input.ty.is_tensor() {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{name}: {:?}", input.ty),
                });
            }
        }

        // Validate that input tensor and corresponding zero_point have the same type.
        // NOTE: This indirectly validates the dtypes of `tensor`.
        for (tensor, tensor_name, zero_point, zero_point_name) in [
            (a, "a", a_zero_point, "a_zero_point"),
            (b, "b", b_zero_point, "b_zero_point"),
        ] {
            let tensor_dtype = tensor.ty.elem_type();
            let zero_point_dtype = zero_point.ty.elem_type();
            if tensor_dtype != zero_point_dtype {
                return Err(ProcessError::TypeMismatch {
                    expected: "Same types for tensor and zero_point".to_string(),
                    actual: format!(
                        "{tensor_name} ({tensor_dtype:?}) vs {zero_point_name} ({zero_point_dtype:?})"
                    ),
                });
            }
        }

        // Validate dtypes for scale inputs
        for (input, name) in [
            (a_scale, "a_scale"),
            (b_scale, "b_scale"),
            (y_scale, "y_scale"),
        ] {
            let dtype = input.ty.elem_type();
            if opset >= 21 {
                if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Only BF16, F16, and F32 dtypes are supported".to_string(),
                        actual: format!("{name}: {dtype:?}"),
                    });
                }
            } else if !matches!(dtype, DType::F32) {
                return Err(ProcessError::TypeMismatch {
                    expected: "Only F32 is supported".to_string(),
                    actual: format!("{name}: {dtype:?}"),
                });
            }
        }

        // NOTE: The shapes of the input tensors are not known at this point,
        //       and so cannot be validated.
        //       However, their ranks can be validated.
        let scale_zp_pairs = [
            (a_scale, a_zero_point, "a"),
            (b_scale, b_zero_point, "b"),
            (y_scale, y_zero_point, "y"),
        ];
        for (scale, zero_point, name) in scale_zp_pairs {
            if scale.ty.rank() != zero_point.ty.rank() {
                return Err(ProcessError::TypeMismatch {
                    expected: format!("{name}_scale and {name}_zero_point must have the same rank"),
                    actual: format!(
                        "{name}_scale rank {} vs {name}_zero_point rank {}",
                        scale.ty.rank(),
                        zero_point.ty.rank()
                    ),
                });
            }
        }

        // Validate ranks for scale and zero point against corresponding input tensor
        for (tensor, scale, name) in [(a, a_scale, "a"), (b, b_scale, "b")] {
            if scale.ty.rank() > tensor.ty.rank() {
                return Err(ProcessError::TypeMismatch {
                    expected: "The ranks for the scale and zero point must be less than or equal to its corresponding tensor".to_string(),
                    actual: format!("{name} has rank {} while {name}_scale has rank {}", tensor.ty.rank(), scale.ty.rank()),
                });
            }
        }

        // Set the output dtype and rank
        let output_dtype = y_zero_point.ty.elem_type();
        let output_rank = a.ty.rank();
        node.outputs[0].ty = ArgType::Tensor(TensorType::new(output_dtype, output_rank, None));

        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::QLinearMatMul(QLinearMatMulNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::OutputPreferences;

    #[test]
    fn test_infer_types_basic() {
        let _ = QLinearMatMulProcessor;
        let _ = OutputPreferences::new();
        todo!("construct test node and assert output dtype/rank once infer_types is implemented")
    }
}
