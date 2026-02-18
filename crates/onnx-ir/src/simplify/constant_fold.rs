use std::cell::RefCell;
use std::rc::Rc;

use crate::TensorDataExt;
use crate::graph_state::GraphState;
use crate::ir::{ArgType, Argument, DType, NodeType, RawNode, TensorData, ValueSource};
use crate::tensor_store::TensorDataRef;

/// Fold nodes where all non-optional inputs are constants.
///
/// Evaluates the operation at compile time and replaces the node with a Constant.
/// Runs after `simplify_constant_shape` so that Shape->Gather results are available
/// as constants for cascading folds (e.g., `Mul(const_3, const_4) -> const_12`).
pub(crate) fn fold_constants(
    mut nodes: Vec<RawNode>,
    graph_outputs: &mut [Argument],
    state: &Rc<RefCell<GraphState>>,
) -> Vec<RawNode> {
    let mut constant_outputs: Vec<String> = Vec::new();

    for node in nodes.iter_mut() {
        // Skip nodes that are already constants
        if node.node_type == NodeType::Constant {
            continue;
        }

        // Check: all non-optional inputs must have constant values
        let all_const = node
            .inputs
            .iter()
            .filter(|arg| !arg.is_optional())
            .all(|arg| arg.value().is_some());

        if !all_const || node.inputs.iter().all(|arg| arg.is_optional()) {
            continue;
        }

        // Try to evaluate the operation
        let (data, output_ty) = match try_evaluate(node) {
            Some(r) => r,
            None => continue,
        };

        let output_name = node.outputs[0].name.clone();

        log::info!(
            "Constant folding: replacing {:?} '{}' with constant",
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

/// Evaluate Concat on constant inputs along the axis attribute.
fn eval_concat(node: &RawNode) -> Option<(TensorData, ArgType)> {
    // Concat only supports axis=0 for 1D constant arrays (shape assembly pattern)
    let axis = node
        .attrs
        .get("axis")
        .map(|v| v.clone().into_i64())
        .unwrap_or(0);
    if axis != 0 {
        return None;
    }

    let output_ty = node.outputs[0].ty.clone();
    let first_data = node.inputs[0].value()?;
    let dtype = first_data.dtype;
    let shape = output_shape(&output_ty);

    let non_optional = || node.inputs.iter().filter(|arg| !arg.is_optional());

    match dtype {
        DType::I64 => {
            let mut result: Vec<i64> = Vec::new();
            for input in non_optional() {
                result.extend(input.value()?.to_i64_vec().ok()?);
            }
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::I32 => {
            let mut result: Vec<i32> = Vec::new();
            for input in non_optional() {
                let vals = input.value()?.to_i64_vec().ok()?;
                result.extend(vals.iter().map(|&v| v as i32));
            }
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::F32 => {
            let mut result: Vec<f32> = Vec::new();
            for input in non_optional() {
                let vals = input.value()?.to_f64_vec().ok()?;
                result.extend(vals.iter().map(|&v| v as f32));
            }
            Some((TensorData::new(result, shape), output_ty))
        }
        DType::F64 => {
            let mut result: Vec<f64> = Vec::new();
            for input in non_optional() {
                result.extend(input.value()?.to_f64_vec().ok()?);
            }
            Some((TensorData::new(result, shape), output_ty))
        }
        _ => None,
    }
}

/// Evaluate Unsqueeze/Squeeze/Reshape by reshaping the constant data
/// to match the already-inferred output type.
fn eval_reshape(node: &RawNode) -> Option<(TensorData, ArgType)> {
    let data = node.inputs[0].value()?;
    let output_ty = node.outputs[0].ty.clone();

    // Determine target shape from output type
    let target_shape = match &output_ty {
        ArgType::Tensor(t) => {
            let static_shape = t.static_shape.as_ref()?;
            // All dims must be known
            let dims: Option<Vec<usize>> = static_shape.iter().map(|d| d.map(|v| v)).collect();
            dims?
        }
        ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => vec![],
        ArgType::Shape(len) => vec![*len],
    };

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

    // Reuse the same bytes, just change shape
    let result = TensorData::from_bytes_vec(data.bytes.to_vec(), target_shape, data.dtype);
    Some((result, output_ty))
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
}
