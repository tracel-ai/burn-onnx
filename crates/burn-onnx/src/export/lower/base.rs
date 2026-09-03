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

/// Convert a range normalized by Burn's slice API into ONNX bounds.
///
/// At capture time Burn has already clamped the ascending interval to the
/// input shape. A negative step reverses that normalized interval; it is not a
/// raw Python-style descending slice. `Tensor::slice` currently returns an
/// empty tensor before the backend operation is recorded, so an empty interval
/// never reaches the captured IR. The branch keeps the conversion total in case
/// that changes.
fn onnx_slice_bounds(start: isize, end: Option<isize>, step: isize) -> (i64, i64) {
    if step > 0 {
        return (start as i64, end.map_or(i64::MAX, |end| end as i64));
    }
    if end.is_some_and(|end| start >= end) {
        // Equal in-range ONNX bounds preserve Burn's empty interval.
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
    let constant_dtype = constant_of_shape_dtype(creation.out.dtype).ok_or_else(|| {
        ExportError::UnsupportedDType {
            tensor: creation.out.id,
            dtype: format!("{:?}", creation.out.dtype),
        }
    })?;
    let constant_output = if constant_dtype == creation.out.dtype {
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
        scalar_tensor(
            constant_dtype,
            ScalarIr::new(value, &constant_dtype),
            creation.out.id,
        )?,
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

/// Return the dtype used for the `ConstantOfShape` value attribute.
///
/// Opset 18 uses the version 9 `ConstantOfShape` schema. `BF16` outputs are
/// created as `F32` and converted by the following `Cast` node. Burn-specific
/// dtypes without an ONNX representation are rejected.
fn constant_of_shape_dtype(dtype: DType) -> Option<DType> {
    match dtype {
        DType::F16
        | DType::F32
        | DType::F64
        | DType::I8
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::U8
        | DType::U16
        | DType::U32
        | DType::U64
        | DType::Bool(_) => Some(dtype),
        DType::BF16 => Some(DType::F32),
        DType::Flex32 | DType::QFloat(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::{BoolStore, DType};

    use super::{constant_of_shape_dtype, onnx_slice_bounds};

    #[test]
    fn constant_of_shape_uses_all_opset_18_native_dtypes() {
        let native_dtypes = [
            DType::F16,
            DType::F32,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::Bool(BoolStore::Native),
        ];

        for dtype in native_dtypes {
            assert_eq!(constant_of_shape_dtype(dtype), Some(dtype));
        }
        assert_eq!(constant_of_shape_dtype(DType::BF16), Some(DType::F32));
        assert_eq!(constant_of_shape_dtype(DType::Flex32), None);
    }

    #[test]
    fn negative_step_keeps_empty_intervals_empty() {
        assert_eq!(onnx_slice_bounds(0, Some(0), -1), (0, 0));
        assert_eq!(onnx_slice_bounds(3, Some(1), -2), (0, 0));
    }

    #[test]
    fn negative_step_reverses_a_normalized_full_range() {
        assert_eq!(onnx_slice_bounds(0, Some(5), -1), (4, i64::MIN));
    }
}
