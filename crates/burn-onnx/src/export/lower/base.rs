//! Lowering for Burn base tensor operations.
//!
//! This family contains operations requiring ONNX-specific operands or
//! attributes, including reshape shape expressions and concatenation axes.

use burn::backend::ir::{BaseOperationIr, OperationIr};

use crate::export::ExportError;

use super::{context::LoweringContext, patterns};

pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
    if let Some(pad) = patterns::constant_pad(&context.graph.graph.operations, index) {
        let pads_name = format!("node_{index}_pads");
        context.i64_initializer(pads_name.clone(), &pad.pads);
        let value_name = format!("node_{index}_value");
        context.scalar_initializer(
            value_name.clone(),
            pad.full.out.dtype,
            pad.full.value,
            pad.full.out.id,
        )?;
        let input = context.tensor_name(pad.slice_assign.value.id);
        let output = context.tensor_name(pad.slice_assign.out.id);
        context.node(
            format!("node_{index}"),
            "Pad",
            vec![input, pads_name, value_name],
            vec![output],
        );
        context.string_attribute("mode", "constant");
        return Ok(true);
    }

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
