//! Lowering for Burn base tensor operations.
//!
//! This family contains operations requiring ONNX-specific operands or
//! attributes, including reshape shape expressions and concatenation axes.

use burn::backend::DType;
use burn::backend::ir::{
    BaseOperationIr, CreationOpIr, OperationIr, ScalarIr, SliceOpIr, TensorIr,
};
use burn::backend::tensor::IndexingUpdateOp;

use crate::export::ExportError;

use super::{context::LoweringContext, onnx_dtype_parts, scalar_tensor};

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
        BaseOperationIr::Slice(slice) => {
            lower_slice(context, index, slice);
            Ok(true)
        }
        BaseOperationIr::Select(select) => {
            let inputs = vec![
                context.tensor_name(select.tensor.id),
                lower_indices(context, index, "select", &select.indices)?,
            ];
            let output = context.tensor_name(select.out.id);
            context.node(format!("node_{index}"), "Gather", inputs, vec![output]);
            context.int_attribute("axis", select.dim as i64);
            Ok(true)
        }
        BaseOperationIr::Scatter(scatter) => {
            let inputs = vec![
                context.tensor_name(scatter.tensor.id),
                lower_indices(context, index, "scatter", &scatter.indices)?,
                context.tensor_name(scatter.value.id),
            ];
            let output = context.tensor_name(scatter.out.id);
            context.node(
                format!("node_{index}"),
                "ScatterElements",
                inputs,
                vec![output],
            );
            context.int_attribute("axis", scatter.dim as i64);
            context.string_attribute(
                "reduction",
                match scatter.update {
                    IndexingUpdateOp::Assign => "none",
                    IndexingUpdateOp::Add => "add",
                    IndexingUpdateOp::Mul => "mul",
                    IndexingUpdateOp::Min => "min",
                    IndexingUpdateOp::Max => "max",
                },
            );
            Ok(true)
        }
        BaseOperationIr::Zeros(creation) => {
            lower_creation(context, index, creation, 0)?;
            Ok(true)
        }
        BaseOperationIr::Ones(creation) => {
            lower_creation(context, index, creation, 1)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn lower_indices(
    context: &mut LoweringContext<'_>,
    index: usize,
    kind: &'static str,
    indices: &TensorIr,
) -> Result<String, ExportError> {
    let input = context.tensor_name(indices.id);
    if matches!(indices.dtype, DType::I32 | DType::I64) {
        return Ok(input);
    }
    if !indices.dtype.is_int() && !indices.dtype.is_uint() {
        return Err(ExportError::UnsupportedOperation {
            operation: index,
            kind: format!("{kind} with non-integer {:?} indices", indices.dtype),
        });
    }

    let output = format!("node_{index}_{kind}_indices64");
    context.node(output.clone(), "Cast", vec![input], vec![output.clone()]);
    context.int_attribute("to", onnx_dtype_parts(indices.id, DType::I64)? as i64);
    Ok(output)
}

fn lower_slice(context: &mut LoweringContext<'_>, index: usize, slice: &SliceOpIr) {
    let (starts, ends): (Vec<_>, Vec<_>) = slice
        .ranges
        .iter()
        .map(|range| onnx_slice_bounds(range.start, range.end, range.step))
        .unzip();
    let axes: Vec<i64> = (0..slice.ranges.len() as i64).collect();
    let steps: Vec<i64> = slice.ranges.iter().map(|range| range.step as i64).collect();

    let starts_name = format!("node_{index}_starts");
    context.i64_initializer(starts_name.clone(), &starts);
    let ends_name = format!("node_{index}_ends");
    context.i64_initializer(ends_name.clone(), &ends);
    let axes_name = format!("node_{index}_axes");
    context.i64_initializer(axes_name.clone(), &axes);
    let steps_name = format!("node_{index}_steps");
    context.i64_initializer(steps_name.clone(), &steps);

    let input = context.tensor_name(slice.tensor.id);
    let output = context.tensor_name(slice.out.id);
    context.node(
        format!("node_{index}"),
        "Slice",
        vec![input, starts_name, ends_name, axes_name, steps_name],
        vec![output],
    );
}

fn onnx_slice_bounds(start: isize, end: Option<isize>, step: isize) -> (i64, i64) {
    if step > 0 {
        return (start as i64, end.map_or(i64::MAX, |end| end as i64));
    }
    if end.is_some_and(|end| start >= end) {
        // Burn permits empty intervals and returns no elements regardless of
        // the step. Equal in-range ONNX bounds preserve that cardinality.
        return (0, 0);
    }

    // Burn selects an ascending [start, end) interval and traverses it in
    // reverse. ONNX uses Python-style descending bounds, so swap the interval
    // endpoints and make each one inclusive/exclusive in the other direction.
    let onnx_start = end.map_or(-1, |end| end.saturating_sub(1) as i64);
    let onnx_end = if start == 0 {
        i64::MIN
    } else {
        start.saturating_sub(1) as i64
    };
    (onnx_start, onnx_end)
}

fn lower_creation(
    context: &mut LoweringContext<'_>,
    index: usize,
    creation: &CreationOpIr,
    value: i64,
) -> Result<(), ExportError> {
    let shape_name = context.shape_input(index, creation.out.id)?;
    let output = context.tensor_name(creation.out.id);
    let constant_output = if creation.out.dtype == DType::I64 {
        output.clone()
    } else {
        format!("node_{index}_constant")
    };
    context.node(
        format!("node_{index}_constant"),
        "ConstantOfShape",
        vec![shape_name],
        vec![constant_output.clone()],
    );
    context.tensor_attribute(
        "value",
        scalar_tensor(DType::I64, ScalarIr::Int(value), creation.out.id)?,
    );
    if constant_output != output {
        context.node(
            format!("node_{index}"),
            "Cast",
            vec![constant_output],
            vec![output],
        );
        context.int_attribute(
            "to",
            onnx_dtype_parts(creation.out.id, creation.out.dtype)? as i64,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::onnx_slice_bounds;

    #[test]
    fn negative_step_keeps_empty_intervals_empty() {
        assert_eq!(onnx_slice_bounds(0, Some(0), -1), (0, 0));
        assert_eq!(onnx_slice_bounds(3, Some(1), -2), (0, 0));
    }
}
