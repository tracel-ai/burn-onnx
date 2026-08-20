//! Recognition of decomposed Burn IR patterns used during lowering.
//!
//! These helpers inspect captured operations but do not rewrite the graph.
//! They should disappear as dedicated Burn IR operations replace the
//! decompositions they recognize.

use burn::backend::ir::{
    BaseOperationIr, FullOpIr, NumericOperationIr, OperationIr, SliceAssignOpIr,
};

/// A decomposed constant-padding pattern recognized in captured Burn IR.
pub(crate) struct ConstantPad<'a> {
    pub(crate) full: &'a FullOpIr,
    pub(crate) slice_assign: &'a SliceAssignOpIr,
    pub(crate) pads: Vec<i64>,
}

// TODO: remove once we have a `PadOpIr` in burn.
/// Recognize `full + slice_assign` as one semantic constant-padding operation.
///
/// This is deliberately isolated from protobuf lowering. It can be removed
/// when Burn capture emits a dedicated padding IR operation.
pub(crate) fn constant_pad(operations: &[OperationIr], index: usize) -> Option<ConstantPad<'_>> {
    let slice_assign = match &operations[index] {
        OperationIr::BaseFloat(BaseOperationIr::SliceAssign(operation))
        | OperationIr::BaseInt(BaseOperationIr::SliceAssign(operation)) => operation,
        _ => return None,
    };
    let full = operations[..index]
        .iter()
        .rev()
        .find_map(full_operation)
        .filter(|full| full.out.id == slice_assign.tensor.id)?;
    let rank = slice_assign.tensor.shape.num_dims();
    let mut starts = Vec::with_capacity(rank);
    let mut ends = Vec::with_capacity(rank);
    if slice_assign.ranges.len() != rank {
        return None;
    }
    for (axis, range) in slice_assign.ranges.iter().enumerate() {
        let end = range.end?;
        if range.step != 1 || range.start < 0 || end < range.start {
            return None;
        }
        let start = range.start as usize;
        let end = end as usize;
        if end > slice_assign.tensor.shape[axis] || end - start != slice_assign.value.shape[axis] {
            return None;
        }
        starts.push(start as i64);
        ends.push((slice_assign.tensor.shape[axis] - end) as i64);
    }
    Some(ConstantPad {
        full,
        slice_assign,
        pads: starts.into_iter().chain(ends).collect(),
    })
}

/// Return whether a `Full` operation is consumed by a recognized constant pad.
pub(crate) fn is_constant_pad_full(operations: &[OperationIr], index: usize) -> bool {
    let Some(full) = full_operation(&operations[index]) else {
        return false;
    };
    let mut consumers = operations
        .iter()
        .enumerate()
        .filter(|(_, operation)| operation.inputs().any(|input| input.id == full.out.id));
    let Some((consumer, _)) = consumers.next() else {
        return false;
    };
    consumers.next().is_none()
        && constant_pad(operations, consumer).is_some_and(|pad| pad.full.out.id == full.out.id)
}

fn full_operation(operation: &OperationIr) -> Option<&FullOpIr> {
    match operation {
        OperationIr::NumericFloat(_, NumericOperationIr::Full(full))
        | OperationIr::NumericInt(_, NumericOperationIr::Full(full)) => Some(full),
        _ => None,
    }
}
