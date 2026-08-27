//! Lowering for Burn base tensor operations.
//!
//! This family contains operations requiring ONNX-specific operands or
//! attributes, including reshape shape expressions and concatenation axes.

use burn::backend::ir::{BaseOperationIr, OperationIr};

use crate::export::ExportError;

use super::context::LoweringContext;

pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
    let base = match operation {
        OperationIr::BaseFloat(operation)
        | OperationIr::BaseInt(operation)
        | OperationIr::BaseBool(operation) => operation,
        _ => return Ok(false),
    };
    match base {
        BaseOperationIr::Reshape(reshape) => {
            let shape_name = context.shape_input(index, reshape.out.id)?;
            let input = context.tensor_name(reshape.input.id);
            let output = context.tensor_name(reshape.out.id);
            context.node(
                format!("node_{index}"),
                "Reshape",
                vec![input, shape_name],
                vec![output],
            );
            Ok(true)
        }
        BaseOperationIr::Cat(cat) => {
            let inputs = cat
                .tensors
                .iter()
                .map(|tensor| context.tensor_name(tensor.id))
                .collect();
            let output = context.tensor_name(cat.out.id);
            context.node(format!("node_{index}"), "Concat", inputs, vec![output]);
            context.int_attribute("axis", cat.dim as i64);
            Ok(true)
        }
        _ => Ok(false),
    }
}
