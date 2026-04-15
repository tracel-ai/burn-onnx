use std::cell::RefCell;
use std::rc::Rc;

use crate::TensorDataExt;
use crate::graph_state::GraphState;
use crate::ir::{ArgType, Argument, DType, NodeType, RawNode, TensorData, ValueSource};
use crate::tensor_store::TensorDataRef;

/// Fold only Slice, Concat, Unsqueeze, Squeeze, and Reshape operations on constant inputs.
///
/// This is a targeted pass designed to run before type inference to resolve weight
/// rearrangement patterns (e.g., PyTorch's GRU/LSTM export inserts Slice+Concat+Unsqueeze
/// chains on initializer weights). Unlike the full `fold_constants`, this avoids folding
/// arithmetic ops (Add, Mul, Div, Sqrt, etc.) that may be needed by pattern matchers like
/// the attention coalescer.
pub(crate) fn fold_weight_rearrangements(
    nodes: Vec<RawNode>,
    state: &Rc<RefCell<GraphState>>,
) -> Vec<RawNode> {
    let filter = |nt: &NodeType| {
        matches!(
            nt,
            NodeType::Slice
                | NodeType::Concat
                | NodeType::Unsqueeze
                | NodeType::Squeeze
                | NodeType::Reshape
        )
    };
    fold_matching(
        nodes,
        &mut [],
        state,
        Some(&filter),
        "Early constant folding",
    )
}

/// Fold nodes where all non-optional inputs are constants.
///
/// Evaluates the operation at compile time and replaces the node with a Constant.
/// Runs after `simplify_constant_shape` so that Shape->Gather results are available
/// as constants for cascading folds (e.g., `Mul(const_3, const_4) -> const_12`).
pub(crate) fn fold_constants(
    nodes: Vec<RawNode>,
    graph_outputs: &mut [Argument],
    state: &Rc<RefCell<GraphState>>,
) -> Vec<RawNode> {
    fold_matching(nodes, graph_outputs, state, None, "Constant folding")
}

/// Shared implementation for constant folding passes.
///
/// When `node_filter` is `Some`, only nodes whose type passes the filter are considered.
/// When `None`, all node types are eligible.
fn fold_matching(
    mut nodes: Vec<RawNode>,
    graph_outputs: &mut [Argument],
    state: &Rc<RefCell<GraphState>>,
    node_filter: Option<&dyn Fn(&NodeType) -> bool>,
    log_prefix: &str,
) -> Vec<RawNode> {
    let mut constant_outputs: Vec<String> = Vec::new();

    for node in nodes.iter_mut() {
        if node.node_type == NodeType::Constant {
            continue;
        }

        if let Some(filter) = node_filter
            && !filter(&node.node_type)
        {
            continue;
        }

        let all_const = node
            .inputs
            .iter()
            .filter(|arg| !arg.is_optional())
            .all(|arg| arg.value().is_some());

        if !all_const || node.inputs.iter().all(|arg| arg.is_optional()) {
            continue;
        }

        let (data, output_ty) = match try_evaluate(node) {
            Some(r) => r,
            None => continue,
        };

        let output_name = node.outputs[0].name.clone();

        log::info!(
            "{log_prefix}: replacing {:?} '{}' with constant",
            node.node_type,
            node.name,
        );

        *node = make_constant_node(&node.name.clone(), &output_name, data, output_ty, state);
        constant_outputs.push(output_name);
    }

    if !constant_outputs.is_empty() {
        super::update_constant_references(&mut nodes, graph_outputs, &constant_outputs);
    }

    nodes
}

/// Create a Constant RawNode from evaluated TensorData.
fn make_constant_node(
    node_name: &str,
    output_name: &str,
    data: TensorData,
    ty: ArgType,
    state: &Rc<RefCell<GraphState>>,
) -> RawNode {
    let data_ref = TensorDataRef::from(data);

    let mut gs = state.borrow_mut();
    let data_id = gs.register_constant(output_name.to_string(), data_ref);
    let value_store = gs.build_value_store();

    let input_name = format!("{}_const", output_name);

    RawNode {
        node_type: NodeType::Constant,
        name: node_name.to_string(),
        inputs: vec![Argument {
            name: input_name,
            ty: ty.clone(),
            value_source: ValueSource::Static(data_id),
            value_store: Some(value_store.clone()),
        }],
        outputs: vec![Argument {
            name: output_name.to_string(),
            ty,
            value_source: ValueSource::Constant,
            value_store: Some(value_store),
        }],
        attrs: std::collections::HashMap::new(),
    }
}

/// Try to evaluate a node with all-constant inputs.
/// Returns the result TensorData and output ArgType, or None if unsupported.
fn try_evaluate(node: &RawNode) -> Option<(TensorData, ArgType)> {
    match node.node_type {
        NodeType::Add => eval_binary(node, BinaryOp::Add),
        NodeType::Sub => eval_binary(node, BinaryOp::Sub),
        NodeType::Mul => eval_binary(node, BinaryOp::Mul),
        NodeType::Div => eval_binary(node, BinaryOp::Div),
        NodeType::Neg => eval_neg(node),
        NodeType::Sqrt => eval_sqrt(node),
        NodeType::Cast => eval_cast(node),
        NodeType::Slice => eval_slice(node),
        NodeType::Concat => eval_concat(node),
        NodeType::Unsqueeze | NodeType::Squeeze | NodeType::Reshape => eval_reshape(node),
        _ => None,
    }
}

enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Evaluate a binary arithmetic operation on constant inputs.
///
/// Supports scalar<->tensor broadcasting (scalar is broadcast to match tensor length)
/// and same-shape element-wise operations. No general multidimensional broadcasting.
fn eval_binary(node: &RawNode, op: BinaryOp) -> Option<(TensorData, ArgType)> {
    if node.inputs.len() < 2 {
        return None;
    }
    let lhs_data = node.inputs[0].value()?;
    let rhs_data = node.inputs[1].value()?;
    let output_ty = node.outputs[0].ty.clone();

    let dtype = lhs_data.dtype;

    let shape = output_shape(&output_ty);

    match dtype {
        DType::I64 => {
            let lhs = lhs_data.to_i64_vec().ok()?;
            let rhs = rhs_data.to_i64_vec().ok()?;
            let result = apply_binary_i64(&lhs, &rhs, &op)?;
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::I32 => {
            let lhs = lhs_data.to_i64_vec().ok()?;
            let rhs = rhs_data.to_i64_vec().ok()?;
            let result_i64 = apply_binary_i64(&lhs, &rhs, &op)?;
            let result: Vec<i32> = result_i64.iter().map(|&v| v as i32).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::F32 => {
            let lhs = lhs_data.to_f64_vec().ok()?;
            let rhs = rhs_data.to_f64_vec().ok()?;
            let result_f64 = apply_binary_f64(&lhs, &rhs, &op)?;
            let result: Vec<f32> = result_f64.iter().map(|&v| v as f32).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::F64 => {
            let lhs = lhs_data.to_f64_vec().ok()?;
            let rhs = rhs_data.to_f64_vec().ok()?;
            let result = apply_binary_f64(&lhs, &rhs, &op)?;
            Some((TensorData::new(result, shape), output_ty))
        }
        _ => None,
    }
}

/// Derive the output shape from the already-inferred output type.
fn output_shape(output_ty: &ArgType) -> Vec<usize> {
    match output_ty {
        ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => vec![],
        ArgType::Shape(len) => vec![*len],
        ArgType::Tensor(t) => t
            .static_shape
            .as_ref()
            .and_then(|ss| ss.iter().copied().collect::<Option<Vec<_>>>())
            .unwrap_or_default(),
    }
}

fn broadcast_get<T: Copy>(slice: &[T], i: usize) -> T {
    if slice.len() == 1 { slice[0] } else { slice[i] }
}

fn apply_binary_i64(lhs: &[i64], rhs: &[i64], op: &BinaryOp) -> Option<Vec<i64>> {
    // Only scalar<->tensor or same-shape
    if lhs.len() != rhs.len() && lhs.len() != 1 && rhs.len() != 1 {
        return None;
    }
    let len = lhs.len().max(rhs.len());
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let a = broadcast_get(lhs, i);
        let b = broadcast_get(rhs, i);
        let val = match op {
            BinaryOp::Add => a.checked_add(b)?,
            BinaryOp::Sub => a.checked_sub(b)?,
            BinaryOp::Mul => a.checked_mul(b)?,
            BinaryOp::Div => {
                if b == 0 {
                    return None;
                }
                a / b
            }
        };
        result.push(val);
    }
    Some(result)
}

fn apply_binary_f64(lhs: &[f64], rhs: &[f64], op: &BinaryOp) -> Option<Vec<f64>> {
    if lhs.len() != rhs.len() && lhs.len() != 1 && rhs.len() != 1 {
        return None;
    }
    let len = lhs.len().max(rhs.len());
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let a = broadcast_get(lhs, i);
        let b = broadcast_get(rhs, i);
        let val = match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => {
                if b == 0.0 {
                    return None;
                }
                a / b
            }
        };
        result.push(val);
    }
    Some(result)
}

/// Evaluate unary negation.
fn eval_neg(node: &RawNode) -> Option<(TensorData, ArgType)> {
    let data = node.inputs[0].value()?;
    let output_ty = node.outputs[0].ty.clone();
    let shape = output_shape(&output_ty);

    match data.dtype {
        DType::I64 => {
            let vals = data.to_i64_vec().ok()?;
            let result: Vec<i64> = vals.iter().map(|v| -v).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::I32 => {
            let vals = data.to_i64_vec().ok()?;
            let result: Vec<i32> = vals.iter().map(|&v| (-v) as i32).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::F32 => {
            let vals = data.to_f64_vec().ok()?;
            let result: Vec<f32> = vals.iter().map(|&v| (-v) as f32).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::F64 => {
            let vals = data.to_f64_vec().ok()?;
            let result: Vec<f64> = vals.iter().map(|v| -v).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        _ => None,
    }
}

/// Evaluate square root on a constant float input.
fn eval_sqrt(node: &RawNode) -> Option<(TensorData, ArgType)> {
    let data = node.inputs[0].value()?;
    let output_ty = node.outputs[0].ty.clone();
    let shape = output_shape(&output_ty);

    match data.dtype {
        DType::F32 => {
            let vals = data.to_f64_vec().ok()?;
            let result: Vec<f32> = vals.iter().map(|&v| v.sqrt() as f32).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::F64 => {
            let vals = data.to_f64_vec().ok()?;
            let result: Vec<f64> = vals.iter().map(|v| v.sqrt()).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        _ => None,
    }
}

/// Evaluate Cast by converting constant data to the target dtype from the output type.
fn eval_cast(node: &RawNode) -> Option<(TensorData, ArgType)> {
    let data = node.inputs[0].value()?;
    let output_ty = node.outputs[0].ty.clone();
    let shape = output_shape(&output_ty);

    let target_dtype = match &output_ty {
        ArgType::ScalarTensor(d) | ArgType::ScalarNative(d) => *d,
        ArgType::Tensor(t) => t.dtype,
        ArgType::Shape(_) => DType::I64,
    };

    if data.dtype == target_dtype {
        return Some((data.clone(), output_ty));
    }

    match (data.dtype, target_dtype) {
        // Integer -> Float
        (DType::I64 | DType::I32, DType::F32) => {
            let vals = data.to_i64_vec().ok()?;
            let result: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        (DType::I64 | DType::I32, DType::F64) => {
            let vals = data.to_i64_vec().ok()?;
            let result: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        // Float -> Float
        (DType::F32 | DType::F64, DType::F32) => {
            let vals = data.to_f64_vec().ok()?;
            let result: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        (DType::F32 | DType::F64, DType::F64) => {
            let vals = data.to_f64_vec().ok()?;
            Some((TensorData::new(vals, shape), output_ty))
        }
        // Float -> Integer
        (DType::F32 | DType::F64, DType::I64) => {
            let vals = data.to_f64_vec().ok()?;
            let result: Vec<i64> = vals.iter().map(|&v| v as i64).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        (DType::F32 | DType::F64, DType::I32) => {
            let vals = data.to_f64_vec().ok()?;
            let result: Vec<i32> = vals.iter().map(|&v| v as i32).collect();
            Some((TensorData::new(result, shape), output_ty))
        }
        _ => None,
    }
}

/// Evaluate Slice on a constant input tensor.
///
/// Supports both attribute-based (opset < 10) and input-based (opset >= 10) parameters.
/// Only handles axis-0 slicing with step=1. This is sufficient for PyTorch RNN weight
/// splitting patterns, which always slice along the gate dimension (axis 0).
fn eval_slice(node: &RawNode) -> Option<(TensorData, ArgType)> {
    let data = node.inputs[0].value()?;
    if data.shape.is_empty() {
        return None;
    }

    // Extract slice parameters from either attributes (opset < 10) or inputs (opset >= 10)
    let (starts, ends, axes) = if node.attrs.contains_key("starts") {
        // Opset < 10: parameters are attributes
        let starts = node.attrs.get("starts")?.clone().into_i64s();
        let ends = node.attrs.get("ends")?.clone().into_i64s();
        let axes = node
            .attrs
            .get("axes")
            .map(|v| v.clone().into_i64s())
            .unwrap_or_else(|| (0..starts.len() as i64).collect());
        (starts, ends, axes)
    } else if node.inputs.len() >= 3 {
        // Opset >= 10: parameters are inputs
        let starts_data = node.inputs.get(1)?.value()?;
        let ends_data = node.inputs.get(2)?.value()?;
        let starts = starts_data.to_i64_vec().ok()?;
        let ends = ends_data.to_i64_vec().ok()?;
        let axes = if let Some(axes_input) = node.inputs.get(3) {
            if let Some(axes_data) = axes_input.value() {
                axes_data.to_i64_vec().ok()?
            } else {
                (0..starts.len() as i64).collect()
            }
        } else {
            (0..starts.len() as i64).collect()
        };
        (starts, ends, axes)
    } else {
        return None;
    };

    // Only support single-axis slicing on axis 0 with default step
    if axes.len() != 1 || axes[0] != 0 || starts.is_empty() || ends.is_empty() {
        return None;
    }

    // Check steps (must be 1 or absent)
    if node.attrs.contains_key("steps") {
        let steps = node.attrs.get("steps")?.clone().into_i64s();
        if steps.iter().any(|&s| s != 1) {
            return None;
        }
    } else if let Some(steps_input) = node.inputs.get(4)
        && let Some(steps_data) = steps_input.value()
    {
        let steps = steps_data.to_i64_vec().ok()?;
        if steps.iter().any(|&s| s != 1) {
            return None;
        }
    }

    let dim0 = data.shape[0];
    let start = clamp_index(starts[0], dim0);
    let end = clamp_index(ends[0], dim0);
    if start >= end {
        return None;
    }

    // For axis=0, the data is contiguous in row-major layout
    let row_size: usize = data.shape[1..].iter().product::<usize>().max(1);
    let elem_size = data.dtype.size();
    let byte_start = start * row_size * elem_size;
    let byte_end = end * row_size * elem_size;
    if byte_end > data.bytes.len() {
        return None;
    }
    let sliced_bytes = &data.bytes[byte_start..byte_end];

    let mut output_shape = data.shape.to_vec();
    output_shape[0] = end - start;

    let output_ty = ArgType::Tensor(crate::ir::TensorType {
        dtype: data.dtype,
        rank: data.shape.len(),
        static_shape: Some(output_shape.iter().map(|&d| Some(d)).collect()),
    });

    let result = TensorData::from_bytes_vec(sliced_bytes.to_vec(), output_shape, data.dtype);
    Some((result, output_ty))
}

/// Clamp a slice index per ONNX spec: negative values wrap, then clamp to [0, dim].
fn clamp_index(idx: i64, dim: usize) -> usize {
    let dim = dim as i64;
    let resolved = if idx < 0 { dim + idx } else { idx };
    resolved.clamp(0, dim) as usize
}

/// Evaluate Concat on constant inputs along the axis attribute.
///
/// Supports axis=0 concatenation on tensors of any rank. For axis=0 in row-major
/// layout, concatenation is equivalent to appending the flat byte arrays.
fn eval_concat(node: &RawNode) -> Option<(TensorData, ArgType)> {
    let axis = node
        .attrs
        .get("axis")
        .map(|v| v.clone().into_i64())
        .unwrap_or(0);
    if axis != 0 {
        return None;
    }

    let non_optional = || node.inputs.iter().filter(|arg| !arg.is_optional());

    // Collect all input data and compute output shape from inputs
    let all_data: Vec<TensorData> = non_optional()
        .map(|input| input.value())
        .collect::<Option<Vec<_>>>()?;

    if all_data.is_empty() {
        return None;
    }

    let dtype = all_data[0].dtype;
    let rank = all_data[0].shape.len();
    if rank == 0 {
        return None;
    }

    // Validate all inputs have matching dtype, rank, and non-axis dimensions
    for d in &all_data[1..] {
        if d.dtype != dtype || d.shape.len() != rank || d.shape[1..] != all_data[0].shape[1..] {
            return None;
        }
    }

    // Compute output shape: sum of axis-0 sizes, rest from first input
    let mut output_shape = all_data[0].shape.to_vec();
    let total_axis0: usize = all_data.iter().map(|d| d.shape[0]).sum();
    output_shape[0] = total_axis0;

    // Concatenate raw bytes (axis=0 in row-major = byte append)
    let total_bytes: usize = all_data.iter().map(|d| d.bytes.len()).sum();
    let mut result_bytes = Vec::with_capacity(total_bytes);
    for d in &all_data {
        result_bytes.extend_from_slice(&d.bytes);
    }

    let output_ty = ArgType::Tensor(crate::ir::TensorType {
        dtype,
        rank,
        static_shape: Some(output_shape.iter().map(|&d| Some(d)).collect()),
    });

    let result = TensorData::from_bytes_vec(result_bytes, output_shape, dtype);
    Some((result, output_ty))
}

/// Evaluate Unsqueeze/Squeeze/Reshape by reshaping the constant data.
///
/// First tries the already-inferred output type. If that's not available (e.g., running
/// before type inference), computes the target shape from operation parameters directly.
fn eval_reshape(node: &RawNode) -> Option<(TensorData, ArgType)> {
    let data = node.inputs[0].value()?;

    // Try to get target shape from already-inferred output type
    let target_shape = match &node.outputs[0].ty {
        ArgType::Tensor(t) if t.static_shape.is_some() => {
            let static_shape = t.static_shape.as_ref()?;
            static_shape.iter().copied().collect::<Option<Vec<_>>>()
        }
        ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => Some(vec![]),
        ArgType::Shape(len) => Some(vec![*len]),
        _ => None,
    };

    // If output type doesn't have shape, compute from operation parameters
    let target_shape = target_shape.or_else(|| compute_reshape_target(node, &data))?;

    // Verify element count matches
    let src_elems: usize = if data.shape.is_empty() {
        1
    } else {
        data.shape.iter().product()
    };
    let dst_elems: usize = if target_shape.is_empty() {
        1
    } else {
        target_shape.iter().product()
    };
    if src_elems != dst_elems {
        return None;
    }

    let output_ty = ArgType::Tensor(crate::ir::TensorType {
        dtype: data.dtype,
        rank: target_shape.len(),
        static_shape: Some(target_shape.iter().map(|&d| Some(d)).collect()),
    });

    let result = TensorData::from_bytes_vec(data.bytes.to_vec(), target_shape, data.dtype);
    Some((result, output_ty))
}

/// Compute target shape for Unsqueeze/Squeeze/Reshape from operation parameters.
///
/// This is the fallback path used when output types haven't been inferred yet
/// (e.g., during early constant folding before type inference).
fn compute_reshape_target(node: &RawNode, data: &TensorData) -> Option<Vec<usize>> {
    match node.node_type {
        NodeType::Unsqueeze => {
            // Get axes from attributes (opset < 13) or from input (opset >= 13)
            let axes = if let Some(attr) = node.attrs.get("axes") {
                attr.clone().into_i64s()
            } else if let Some(axes_input) = node.inputs.get(1) {
                axes_input.value()?.to_i64_vec().ok()?
            } else {
                return None;
            };

            let output_rank = data.shape.len() + axes.len();
            let output_rank_i64 = output_rank as i64;
            let mut result = data.shape.to_vec();
            let mut sorted_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    let normalized = if a < 0 { output_rank_i64 + a } else { a };
                    if normalized < 0 || normalized >= output_rank_i64 {
                        None
                    } else {
                        Some(normalized as usize)
                    }
                })
                .collect::<Option<Vec<_>>>()?;
            sorted_axes.sort();
            for &ax in &sorted_axes {
                if ax > result.len() {
                    return None;
                }
                result.insert(ax, 1);
            }
            Some(result)
        }
        NodeType::Squeeze => {
            // Get axes from attributes (opset < 13) or from input (opset >= 13)
            let axes = if let Some(attr) = node.attrs.get("axes") {
                attr.clone().into_i64s()
            } else if let Some(axes_input) = node.inputs.get(1) {
                axes_input.value()?.to_i64_vec().ok()?
            } else {
                // Default: squeeze all dims with size 1
                let squeezed: Vec<usize> = data.shape.iter().copied().filter(|&d| d != 1).collect();
                return Some(squeezed);
            };

            let rank = data.shape.len() as i64;
            let mut axes_set: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (rank + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            axes_set.sort();
            Some(
                data.shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !axes_set.contains(i))
                    .map(|(_, &d)| d)
                    .collect(),
            )
        }
        NodeType::Reshape => {
            // Read target shape from the second input (opset >= 5)
            let shape_data = node.inputs.get(1)?.value()?;
            let shape_vals = shape_data.to_i64_vec().ok()?;
            let src_elems: usize = if data.shape.is_empty() {
                1
            } else {
                data.shape.iter().product()
            };
            // Resolve -1 dimension (at most one allowed)
            let mut result = Vec::with_capacity(shape_vals.len());
            let mut neg_idx = None;
            let mut known_product: usize = 1;
            for (i, &v) in shape_vals.iter().enumerate() {
                if v == -1 {
                    if neg_idx.is_some() {
                        return None; // multiple -1s
                    }
                    neg_idx = Some(i);
                    result.push(0); // placeholder
                } else if v == 0 {
                    // 0 means "copy from input" per ONNX spec
                    let dim = *data.shape.get(i)?;
                    known_product *= dim;
                    result.push(dim);
                } else if v > 0 {
                    known_product *= v as usize;
                    result.push(v as usize);
                } else {
                    return None; // invalid negative value
                }
            }
            if let Some(idx) = neg_idx {
                if known_product == 0 {
                    return None;
                }
                result[idx] = src_elems / known_product;
            }
            Some(result)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::AttributeValue;
    use crate::simplify::tests::arg;
    use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};

    fn test_state() -> Rc<RefCell<GraphState>> {
        Rc::new(RefCell::new(GraphState::new(&[], &[], &[], &[])))
    }

    fn const_i64_scalar(name: &str, value: i64) -> Argument {
        Argument::from_const_i64(name, value)
    }

    fn const_i64_vec(name: &str, values: &[i64]) -> Argument {
        Argument::from_const_i64_shape(name, values)
    }

    fn const_f32_scalar(name: &str, value: f32) -> Argument {
        let bytes = bytes::Bytes::copy_from_slice(&value.to_ne_bytes());
        let data_ref = TensorDataRef::new(bytes, vec![], DType::F32);
        let mut store = TensorStore::new();
        let id = store.store(data_ref);
        let mut constant_map = std::collections::HashMap::new();
        constant_map.insert(name.to_string(), id);
        let value_store = ValueStore::new(
            std::sync::Arc::new(store),
            std::sync::Arc::new(constant_map),
        );
        Argument {
            name: name.to_string(),
            ty: ArgType::ScalarNative(DType::F32),
            value_source: ValueSource::Constant,
            value_store: Some(value_store),
        }
    }

    fn raw_node(
        name: &str,
        node_type: NodeType,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    ) -> RawNode {
        RawNode {
            node_type,
            name: name.to_string(),
            inputs,
            outputs,
            attrs: std::collections::HashMap::new(),
        }
    }

    fn raw_node_with_attrs(
        name: &str,
        node_type: NodeType,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        attrs: std::collections::HashMap<String, AttributeValue>,
    ) -> RawNode {
        RawNode {
            node_type,
            name: name.to_string(),
            inputs,
            outputs,
            attrs,
        }
    }

    fn scalar_out(name: &str, dtype: DType) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::ScalarNative(dtype),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn shape_out(name: &str, len: usize) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Shape(len),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    #[test]
    fn test_add_i64_constants_folded() {
        let nodes = vec![raw_node(
            "add",
            NodeType::Add,
            vec![const_i64_scalar("a", 3), const_i64_scalar("b", 4)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        let n = &result[0];
        assert_eq!(n.node_type, NodeType::Constant);
        let val = n.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 7);
    }

    #[test]
    fn test_mul_i64_constants_folded() {
        let nodes = vec![raw_node(
            "mul",
            NodeType::Mul,
            vec![const_i64_scalar("a", 3), const_i64_scalar("b", 4)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        let n = &result[0];
        assert_eq!(n.node_type, NodeType::Constant);
        let val = n.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 12);
    }

    #[test]
    fn test_div_by_zero_not_folded() {
        let nodes = vec![raw_node(
            "div",
            NodeType::Div,
            vec![const_i64_scalar("a", 10), const_i64_scalar("b", 0)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Div);
    }

    #[test]
    fn test_binary_f32_folded() {
        let nodes = vec![raw_node(
            "add",
            NodeType::Add,
            vec![const_f32_scalar("a", 1.5), const_f32_scalar("b", 2.5)],
            vec![scalar_out("out", DType::F32)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        let n = &result[0];
        assert_eq!(n.node_type, NodeType::Constant);
        let val = n.inputs[0].value().unwrap().scalar_f32().unwrap();
        assert!((val - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_broadcast_folded() {
        // scalar(2) * [3, 4, 5] -> [6, 8, 10]
        let nodes = vec![raw_node(
            "mul",
            NodeType::Mul,
            vec![const_i64_scalar("a", 2), const_i64_vec("b", &[3, 4, 5])],
            vec![shape_out("out", 3)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        let n = &result[0];
        assert_eq!(n.node_type, NodeType::Constant);
        let vals = n.inputs[0].value().unwrap().to_i64_vec().unwrap();
        assert_eq!(vals, vec![6, 8, 10]);
    }

    #[test]
    fn test_neg_folded() {
        let nodes = vec![raw_node(
            "neg",
            NodeType::Neg,
            vec![const_i64_scalar("a", 5)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        let n = &result[0];
        assert_eq!(n.node_type, NodeType::Constant);
        let val = n.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, -5);
    }

    #[test]
    fn test_concat_folded() {
        let nodes = vec![raw_node_with_attrs(
            "concat",
            NodeType::Concat,
            vec![const_i64_vec("a", &[1, 2]), const_i64_vec("b", &[3, 4, 5])],
            vec![shape_out("out", 5)],
            [("axis".to_string(), AttributeValue::Int64(0))]
                .into_iter()
                .collect(),
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        let n = &result[0];
        assert_eq!(n.node_type, NodeType::Constant);
        let vals = n.inputs[0].value().unwrap().to_i64_vec().unwrap();
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_dynamic_input_not_folded() {
        let nodes = vec![raw_node(
            "add",
            NodeType::Add,
            vec![arg("dynamic_x"), const_i64_scalar("b", 4)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Add);
    }

    #[test]
    fn test_downstream_refs_updated() {
        // Mul(const_3, const_4) -> const_12, then a downstream Add uses it
        let nodes = vec![
            raw_node(
                "mul",
                NodeType::Mul,
                vec![const_i64_scalar("a", 3), const_i64_scalar("b", 4)],
                vec![scalar_out("mul_out", DType::I64)],
            ),
            raw_node(
                "add",
                NodeType::Add,
                vec![scalar_out("mul_out", DType::I64), arg("x")],
                vec![arg("add_out")],
            ),
        ];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);

        // The mul should be folded
        assert_eq!(result[0].node_type, NodeType::Constant);

        // The add's first input should now be Constant with value_store
        let add_node = &result[1];
        assert_eq!(add_node.inputs[0].value_source, ValueSource::Constant);
        let val = add_node.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 12);
    }

    #[test]
    fn test_cast_i64_to_f32() {
        let nodes = vec![raw_node(
            "cast",
            NodeType::Cast,
            vec![const_i64_scalar("a", 3)],
            vec![scalar_out("out", DType::F32)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Constant);
        let val = result[0].inputs[0].value().unwrap().scalar_f32().unwrap();
        assert!((val - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cast_f32_to_i64() {
        let nodes = vec![raw_node(
            "cast",
            NodeType::Cast,
            vec![const_f32_scalar("a", 7.9)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Constant);
        let val = result[0].inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 7); // truncation
    }

    #[test]
    fn test_cast_same_dtype_noop() {
        let nodes = vec![raw_node(
            "cast",
            NodeType::Cast,
            vec![const_i64_scalar("a", 42)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Constant);
        let val = result[0].inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_sqrt_f32() {
        let nodes = vec![raw_node(
            "sqrt",
            NodeType::Sqrt,
            vec![const_f32_scalar("a", 9.0)],
            vec![scalar_out("out", DType::F32)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Constant);
        let val = result[0].inputs[0].value().unwrap().scalar_f32().unwrap();
        assert!((val - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt_i64_not_folded() {
        // Sqrt on integer is not supported
        let nodes = vec![raw_node(
            "sqrt",
            NodeType::Sqrt,
            vec![const_i64_scalar("a", 9)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Sqrt);
    }

    /// Helper to create a constant f32 tensor argument with a given shape.
    fn const_f32_tensor(name: &str, values: &[f32], shape: Vec<usize>) -> Argument {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let data_ref = TensorDataRef::new(
            bytes::Bytes::copy_from_slice(&bytes),
            shape.clone(),
            DType::F32,
        );
        let mut store = TensorStore::new();
        let id = store.store(data_ref);
        let mut constant_map = std::collections::HashMap::new();
        constant_map.insert(name.to_string(), id);
        let value_store = ValueStore::new(
            std::sync::Arc::new(store),
            std::sync::Arc::new(constant_map),
        );
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(crate::ir::TensorType {
                dtype: DType::F32,
                rank: shape.len(),
                static_shape: Some(shape.iter().map(|&d| Some(d)).collect()),
            }),
            value_source: ValueSource::Constant,
            value_store: Some(value_store),
        }
    }

    /// Helper to create a dynamic tensor output argument.
    fn dynamic_tensor_out(name: &str, dtype: DType, rank: usize) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(crate::ir::TensorType {
                dtype,
                rank,
                static_shape: None,
            }),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    #[test]
    fn test_slice_axis0_2d_attr() {
        // Slice a [4, 2] f32 tensor along axis 0 from index 1 to 3
        // Input: [[1,2], [3,4], [5,6], [7,8]]
        // Expected output: [[3,4], [5,6]]
        let input = const_f32_tensor("w", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);

        let nodes = vec![raw_node_with_attrs(
            "slice",
            NodeType::Slice,
            vec![input],
            vec![dynamic_tensor_out("out", DType::F32, 2)],
            [
                ("starts".to_string(), AttributeValue::Int64s(vec![1])),
                ("ends".to_string(), AttributeValue::Int64s(vec![3])),
                ("axes".to_string(), AttributeValue::Int64s(vec![0])),
            ]
            .into_iter()
            .collect(),
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Constant);

        let data = result[0].inputs[0].value().unwrap();
        assert_eq!(data.shape.to_vec(), vec![2, 2]);
        let vals = data.to_f64_vec().unwrap();
        assert_eq!(vals, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_slice_then_concat_then_unsqueeze_cascade() {
        // Simulates PyTorch GRU weight rearrangement pattern:
        // Slice constant -> Concat sliced results -> Unsqueeze
        let input = const_f32_tensor("w", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6, 1]);
        let state = test_state();

        let nodes = vec![
            raw_node_with_attrs(
                "slice1",
                NodeType::Slice,
                vec![input.clone()],
                vec![dynamic_tensor_out("slice_out", DType::F32, 2)],
                [
                    ("starts".to_string(), AttributeValue::Int64s(vec![3])),
                    ("ends".to_string(), AttributeValue::Int64s(vec![6])),
                    ("axes".to_string(), AttributeValue::Int64s(vec![0])),
                ]
                .into_iter()
                .collect(),
            ),
            raw_node_with_attrs(
                "slice2",
                NodeType::Slice,
                vec![input],
                vec![dynamic_tensor_out("slice2_out", DType::F32, 2)],
                [
                    ("starts".to_string(), AttributeValue::Int64s(vec![0])),
                    ("ends".to_string(), AttributeValue::Int64s(vec![3])),
                    ("axes".to_string(), AttributeValue::Int64s(vec![0])),
                ]
                .into_iter()
                .collect(),
            ),
            raw_node_with_attrs(
                "concat1",
                NodeType::Concat,
                vec![
                    dynamic_tensor_out("slice_out", DType::F32, 2),
                    dynamic_tensor_out("slice2_out", DType::F32, 2),
                ],
                vec![dynamic_tensor_out("concat_out", DType::F32, 2)],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
            raw_node_with_attrs(
                "unsqueeze1",
                NodeType::Unsqueeze,
                vec![dynamic_tensor_out("concat_out", DType::F32, 2)],
                vec![dynamic_tensor_out("unsqueeze_out", DType::F32, 3)],
                [("axes".to_string(), AttributeValue::Int64s(vec![0]))]
                    .into_iter()
                    .collect(),
            ),
        ];

        // Run targeted early folding with cascade
        let mut folded = nodes;
        for _ in 0..5 {
            let before = folded
                .iter()
                .filter(|n| n.node_type == NodeType::Constant)
                .count();
            folded = fold_weight_rearrangements(folded, &state);
            if folded
                .iter()
                .filter(|n| n.node_type == NodeType::Constant)
                .count()
                == before
            {
                break;
            }
        }

        // All 4 nodes should be folded to constants
        assert_eq!(folded[0].node_type, NodeType::Constant);
        assert_eq!(folded[1].node_type, NodeType::Constant);
        assert_eq!(folded[2].node_type, NodeType::Constant);
        assert_eq!(folded[3].node_type, NodeType::Constant);

        // Verify the final unsqueeze result: [4,5,6,1,2,3] unsqueezed to [1,6,1]
        let data = folded[3].inputs[0].value().unwrap();
        assert_eq!(data.shape.to_vec(), vec![1, 6, 1]);
        let vals = data.to_f64_vec().unwrap();
        assert_eq!(vals, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_slice_negative_index() {
        // Slice [4, 2] from starts=[-2] ends=[i64::MAX] => rows [2:4]
        let input = const_f32_tensor("w", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2]);
        let nodes = vec![raw_node_with_attrs(
            "slice",
            NodeType::Slice,
            vec![input],
            vec![dynamic_tensor_out("out", DType::F32, 2)],
            [
                ("starts".to_string(), AttributeValue::Int64s(vec![-2])),
                ("ends".to_string(), AttributeValue::Int64s(vec![i64::MAX])),
                ("axes".to_string(), AttributeValue::Int64s(vec![0])),
            ]
            .into_iter()
            .collect(),
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Constant);

        let data = result[0].inputs[0].value().unwrap();
        assert_eq!(data.shape.to_vec(), vec![2, 2]);
        let vals = data.to_f64_vec().unwrap();
        assert_eq!(vals, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_slice_opset10_input_based() {
        // Opset >= 10: starts/ends/axes come from input tensors, not attributes
        let input = const_f32_tensor("w", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let starts = const_i64_vec("starts", &[1]);
        let ends = const_i64_vec("ends", &[3]);
        let axes = const_i64_vec("axes", &[0]);

        let nodes = vec![raw_node(
            "slice",
            NodeType::Slice,
            vec![input, starts, ends, axes],
            vec![dynamic_tensor_out("out", DType::F32, 2)],
        )];

        let state = test_state();
        let result = fold_constants(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Constant);

        let data = result[0].inputs[0].value().unwrap();
        assert_eq!(data.shape.to_vec(), vec![2, 2]);
        let vals = data.to_f64_vec().unwrap();
        assert_eq!(vals, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_weight_rearrangement_skips_arithmetic() {
        // fold_weight_rearrangements should NOT fold Add even with constant inputs
        let nodes = vec![raw_node(
            "add",
            NodeType::Add,
            vec![const_i64_scalar("a", 3), const_i64_scalar("b", 4)],
            vec![scalar_out("out", DType::I64)],
        )];

        let state = test_state();
        let result = fold_weight_rearrangements(nodes, &state);
        assert_eq!(
            result[0].node_type,
            NodeType::Add,
            "arithmetic ops must not be folded"
        );
    }
}
