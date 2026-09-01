//! Lowering for Burn base tensor operations.
//!
//! This family contains operations requiring ONNX-specific operands or
//! attributes, including reshape shape expressions and concatenation axes.

use burn::backend::ir::{BaseOperationIr, CreationOpIr, OperationIr, ScalarIr, SliceOpIr};
use burn::backend::tensor::IndexingUpdateOp;

use crate::export::ExportError;

use super::context::LoweringContext;
use super::scalar_tensor;

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
                context.tensor_name(select.indices.id),
            ];
            let output = context.tensor_name(select.out.id);
            context.node(format!("node_{index}"), "Gather", inputs, vec![output]);
            context.int_attribute("axis", select.dim as i64);
            Ok(true)
        }
        BaseOperationIr::Scatter(scatter) => {
            let inputs = vec![
                context.tensor_name(scatter.tensor.id),
                context.tensor_name(scatter.indices.id),
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
            lower_creation(context, index, creation, 0.0)?;
            Ok(true)
        }
        BaseOperationIr::Ones(creation) => {
            lower_creation(context, index, creation, 1.0)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn lower_slice(context: &mut LoweringContext<'_>, index: usize, slice: &SliceOpIr) {
    let starts: Vec<i64> = slice
        .ranges
        .iter()
        .map(|range| range.start as i64)
        .collect();
    let ends: Vec<i64> = slice
        .ranges
        .iter()
        .map(|range| match range.end {
            Some(end) => end as i64,
            // An open end takes everything the step direction can reach; ONNX
            // clamps out-of-range bounds to the dimension.
            None if range.step >= 0 => i64::MAX,
            None => i64::MIN,
        })
        .collect();
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

fn lower_creation(
    context: &mut LoweringContext<'_>,
    index: usize,
    creation: &CreationOpIr,
    value: f64,
) -> Result<(), ExportError> {
    let shape_name = context.shape_input(index, creation.out.id)?;
    let output = context.tensor_name(creation.out.id);
    context.node(
        format!("node_{index}"),
        "ConstantOfShape",
        vec![shape_name],
        vec![output],
    );
    context.tensor_attribute(
        "value",
        scalar_tensor(
            creation.out.dtype,
            ScalarIr::new(value, &creation.out.dtype),
            creation.out.id,
        )?,
    );
    Ok(())
}
