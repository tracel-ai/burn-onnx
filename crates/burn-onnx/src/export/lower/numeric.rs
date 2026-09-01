//! Lowering for numeric tensor and tensor-scalar operations.
//!
//! Scalar operands and creation shapes are materialized as ONNX initializers
//! before their corresponding nodes are emitted.

use burn::backend::DType;
use burn::backend::ir::{
    NumericOperationIr, OperationIr, PadModeIr, PadOpIr, ReduceDimWithIndicesOpIr, ScalarOpIr,
};

use crate::export::ExportError;

use super::{context::LoweringContext, onnx_dtype_parts, scalar_tensor};

pub(super) fn lower(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<bool, ExportError> {
    let numeric = match operation {
        OperationIr::NumericFloat(_, operation) | OperationIr::NumericInt(_, operation) => {
            operation
        }
        _ => return Ok(false),
    };
    if let NumericOperationIr::Full(full) = numeric {
        let shape_name = context.shape_input(index, full.out.id)?;
        let output = context.tensor_name(full.out.id);
        context.node(
            format!("node_{index}"),
            "ConstantOfShape",
            vec![shape_name],
            vec![output],
        );
        context.tensor_attribute(
            "value",
            scalar_tensor(full.out.dtype, full.value, full.out.id)?,
        );
        return Ok(true);
    }
    if let NumericOperationIr::Pad(pad) = numeric {
        lower_pad(context, index, pad)?;
        return Ok(true);
    }
    if let NumericOperationIr::SumDim(reduce) = numeric {
        let axes_name = format!("node_{index}_axes");
        context.i64_initializer(axes_name.clone(), &[reduce.axis as i64]);
        let input = context.tensor_name(reduce.input.id);
        let output = context.tensor_name(reduce.out.id);
        context.node(
            format!("node_{index}"),
            "ReduceSum",
            vec![input, axes_name],
            vec![output],
        );
        let kept = reduce.out.shape.num_dims() == reduce.input.shape.num_dims();
        context.int_attribute("keepdims", kept as i64);
        return Ok(true);
    }
    if let NumericOperationIr::MaxDimWithIndices(reduce) = numeric {
        lower_reduce_with_indices(context, index, reduce, "ArgMax")?;
        return Ok(true);
    }
    if let NumericOperationIr::MinDimWithIndices(reduce) = numeric {
        lower_reduce_with_indices(context, index, reduce, "ArgMin")?;
        return Ok(true);
    }
    if let Some((op_type, scalar)) = scalar_operation(numeric) {
        lower_scalar(context, index, op_type, scalar)?;
        return Ok(true);
    }
    let op_type = match numeric {
        NumericOperationIr::Add(_) => "Add",
        NumericOperationIr::Sub(_) => "Sub",
        NumericOperationIr::Mul(_) => "Mul",
        NumericOperationIr::Div(_) => "Div",
        NumericOperationIr::Abs(_) => "Abs",
        NumericOperationIr::Neg(_) => "Neg",
        _ => return Ok(false),
    };
    let inputs = operation
        .inputs()
        .map(|tensor| context.tensor_name(tensor.id))
        .collect();
    let outputs = operation
        .outputs()
        .map(|tensor| context.tensor_name(tensor.id))
        .collect();
    context.node(format!("node_{index}"), op_type, inputs, outputs);
    Ok(true)
}

/// Lower a max or min reduction that also yields the winning indices.
///
/// ONNX has no single operator for this pair. `ArgMax`/`ArgMin` produces the
/// indices, and `GatherElements` reads the values back out through them, which
/// matches the reduction exactly instead of recomputing it.
fn lower_reduce_with_indices(
    context: &mut LoweringContext<'_>,
    index: usize,
    reduce: &ReduceDimWithIndicesOpIr,
    arg_op: &'static str,
) -> Result<(), ExportError> {
    // `GatherElements` needs indices of the input's rank, which only the
    // dimension-keeping form provides.
    if reduce.out.shape.num_dims() != reduce.tensor.shape.num_dims() {
        return Err(ExportError::UnsupportedOperation {
            operation: index,
            kind: format!("{arg_op} reduction that drops the reduced dimension"),
        });
    }

    let indices = context.tensor_name(reduce.out_indices.id);
    // `ArgMax`/`ArgMin` always produce `int64`; a cast reconciles the traced
    // index dtype where it differs.
    let indices64 = match reduce.out_indices.dtype {
        DType::I64 => indices.clone(),
        _ => format!("node_{index}_indices64"),
    };
    let input = context.tensor_name(reduce.tensor.id);
    context.node(
        format!("node_{index}_arg"),
        arg_op,
        vec![input.clone()],
        vec![indices64.clone()],
    );
    context.int_attribute("axis", reduce.dim as i64);
    context.int_attribute("keepdims", 1);
    if indices64 != indices {
        context.node(
            format!("node_{index}_cast"),
            "Cast",
            vec![indices64.clone()],
            vec![indices],
        );
        context.int_attribute(
            "to",
            onnx_dtype_parts(reduce.out_indices.id, reduce.out_indices.dtype)? as i64,
        );
    }
    let output = context.tensor_name(reduce.out.id);
    context.node(
        format!("node_{index}"),
        "GatherElements",
        vec![input, indices64],
        vec![output],
    );
    context.int_attribute("axis", reduce.dim as i64);
    Ok(())
}

fn lower_pad(
    context: &mut LoweringContext<'_>,
    index: usize,
    pad: &PadOpIr,
) -> Result<(), ExportError> {
    let pads_name = format!("node_{index}_pads");
    let pads = pad
        .padding
        .iter()
        .map(|(before, _)| *before as i64)
        .chain(pad.padding.iter().map(|(_, after)| *after as i64))
        .collect::<Vec<_>>();
    context.i64_initializer(pads_name.clone(), &pads);

    let mut inputs = vec![context.tensor_name(pad.input.id), pads_name];
    let mode = match pad.mode {
        PadModeIr::Constant(value) => {
            let value_name = format!("node_{index}_value");
            context.scalar_initializer(value_name.clone(), pad.input.dtype, value, pad.input.id)?;
            inputs.push(value_name);
            "constant"
        }
        PadModeIr::Reflect => "reflect",
        PadModeIr::Edge => "edge",
    };
    let output = context.tensor_name(pad.out.id);
    context.node(format!("node_{index}"), "Pad", inputs, vec![output]);
    context.string_attribute("mode", mode);
    Ok(())
}

fn scalar_operation(operation: &NumericOperationIr) -> Option<(&'static str, &ScalarOpIr)> {
    match operation {
        NumericOperationIr::AddScalar(operation) => Some(("Add", operation)),
        NumericOperationIr::SubScalar(operation) => Some(("Sub", operation)),
        NumericOperationIr::MulScalar(operation) => Some(("Mul", operation)),
        NumericOperationIr::DivScalar(operation) => Some(("Div", operation)),
        _ => None,
    }
}

fn lower_scalar(
    context: &mut LoweringContext<'_>,
    index: usize,
    op_type: &'static str,
    scalar: &ScalarOpIr,
) -> Result<(), ExportError> {
    let scalar_name = format!("node_{index}_scalar");
    context.scalar_initializer(
        scalar_name.clone(),
        scalar.lhs.dtype,
        scalar.rhs,
        scalar.lhs.id,
    )?;
    let input = context.tensor_name(scalar.lhs.id);
    let output = context.tensor_name(scalar.out.id);
    context.node(
        format!("node_{index}"),
        op_type,
        vec![input, scalar_name],
        vec![output],
    );
    Ok(())
}
