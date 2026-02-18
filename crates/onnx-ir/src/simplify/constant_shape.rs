use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::TensorDataExt;
use crate::graph_state::GraphState;
use crate::ir::{ArgType, Argument, DType, NodeType, RawNode, ValueSource};
use crate::tensor_store::TensorDataRef;

/// Simplify shape-related patterns when input shapes are statically known.
///
/// Handles two patterns:
/// 1. `Shape -> Gather(constant_index)` -> constant scalar
/// 2. `Shape -> Slice(static starts/ends)` -> constant tensor
///
/// Orphaned nodes are cleaned up by dead node elimination.
pub(crate) fn simplify_constant_shape(
    mut nodes: Vec<RawNode>,
    graph_outputs: &mut [Argument],
    state: &Rc<RefCell<GraphState>>,
) -> Vec<RawNode> {
    // Build output_name -> node index map
    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.name.clone(), i);
        }
    }

    // Pass 1: Shape -> Gather(constant_index) -> constant scalar
    let mut gather_replacements: Vec<(usize, i64)> = Vec::new();
    for (gi, node) in nodes.iter().enumerate() {
        if node.node_type != NodeType::Gather || node.inputs.len() < 2 {
            continue;
        }
        if let Some(dim_value) = extract_constant_shape_dim(node, &nodes, &producer) {
            gather_replacements.push((gi, dim_value));
        }
    }

    // Track all output names replaced with constants so we can update downstream references
    let mut constant_outputs: Vec<String> = Vec::new();

    for (gi, dim_value) in &gather_replacements {
        let gather = &nodes[*gi];
        log::info!(
            "Simplification: replacing Shape->Gather '{}' with constant {}",
            gather.name,
            dim_value,
        );

        let output_name = gather.outputs[0].name.clone();
        let output_ty = gather.outputs[0].ty.clone();
        let node_name = nodes[*gi].name.clone();
        nodes[*gi] = make_constant_node(&node_name, &output_name, &[*dim_value], output_ty, state);
        constant_outputs.push(output_name);
    }

    // Pass 2: Shape -> Slice(static) -> constant tensor
    let mut slice_replacements: Vec<(usize, Vec<i64>)> = Vec::new();
    for (si, node) in nodes.iter().enumerate() {
        if node.node_type != NodeType::Slice || node.inputs.is_empty() {
            continue;
        }
        if let Some(values) = extract_constant_shape_slice(node, &nodes, &producer) {
            slice_replacements.push((si, values));
        }
    }

    for (si, values) in &slice_replacements {
        let slice_node = &nodes[*si];
        log::info!(
            "Simplification: replacing Shape->Slice '{}' with constant {:?}",
            slice_node.name,
            values,
        );

        let output_name = slice_node.outputs[0].name.clone();
        let node_name = nodes[*si].name.clone();
        nodes[*si] = make_constant_node(
            &node_name,
            &output_name,
            values,
            ArgType::Shape(values.len()),
            state,
        );
        constant_outputs.push(output_name);
    }

    // Pass 3 (full Shape elimination) is intentionally omitted. While it works when
    // static_shape values match runtime shapes, type inference populates static_shape
    // from ONNX export-time values which may differ at runtime for models with dynamic
    // spatial dimensions (e.g., rf-detr). The Shape codegen already handles the dynamic
    // case correctly via .dims(), so skipping this pass is safe.

    // Update downstream inputs: nodes that reference replaced outputs need
    // ValueSource::Constant so the dead constant elimination pass in
    // convert_to_graph recognizes them as live references.
    if !constant_outputs.is_empty() {
        super::update_constant_references(&mut nodes, graph_outputs, &constant_outputs);
    }

    nodes
}

/// Create a Constant RawNode with its data registered in GraphState.
fn make_constant_node(
    node_name: &str,
    output_name: &str,
    values: &[i64],
    ty: ArgType,
    state: &Rc<RefCell<GraphState>>,
) -> RawNode {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let shape = if values.len() == 1 && matches!(ty, ArgType::ScalarNative(_)) {
        vec![]
    } else {
        vec![values.len()]
    };
    let data_ref = TensorDataRef::new(bytes::Bytes::from(bytes), shape, DType::I64);

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
        attrs: HashMap::new(),
    }
}

/// Check if a Gather node reads from a Shape node with a statically known input,
/// and if so, return the constant dimension value.
fn extract_constant_shape_dim(
    gather: &RawNode,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
) -> Option<i64> {
    // Only fold when the output is a scalar or single-element shape.
    // Gather with tensor indices produces Shape(N) outputs that can't be
    // represented as a single constant value.
    match &gather.outputs[0].ty {
        ArgType::ScalarNative(_) => {}
        ArgType::Shape(1) => {}
        _ => return None,
    }

    // Gather's axis must be 0 (indexing into the 1D shape output)
    let axis = gather
        .attrs
        .get("axis")
        .map(|v| v.clone().into_i64())
        .unwrap_or(0);
    if axis != 0 {
        return None;
    }

    // Gather's data input (input[0]) must come from a Shape node
    let shape_idx = *producer.get(&gather.inputs[0].name)?;
    let shape_node = &nodes[shape_idx];
    if shape_node.node_type != NodeType::Shape {
        return None;
    }

    // The Shape node's input must have a fully known static shape (all dims concrete)
    let static_shape = shape_node.inputs[0].ty.static_shape_known()?;

    // Account for Shape's start/end attributes (opset 15+)
    let rank = static_shape.len();
    let mut start = shape_node
        .attrs
        .get("start")
        .map(|v| v.clone().into_i64())
        .unwrap_or(0);
    if start < 0 {
        start += rank as i64;
    }
    let start = start.max(0) as usize;

    // Gather's index (input[1]) must be a constant scalar
    let mut index_val = gather.inputs[1].value()?.scalar_i64().ok()?;

    // Handle negative indices (relative to the sliced shape length)
    let sliced_len = static_shape.len() - start;
    if index_val < 0 {
        index_val += sliced_len as i64;
    }
    if index_val < 0 || index_val as usize >= sliced_len {
        return None;
    }

    let dim_idx = start + index_val as usize;
    Some(static_shape[dim_idx] as i64)
}

/// Extract the full static shape from a Shape node, accounting for start/end attributes.
fn extract_full_static_shape(shape_node: &RawNode) -> Option<Vec<i64>> {
    let static_shape = shape_node.inputs[0].ty.static_shape_known()?;
    let rank = static_shape.len();

    let mut start = shape_node
        .attrs
        .get("start")
        .map(|v| v.clone().into_i64())
        .unwrap_or(0);
    if start < 0 {
        start += rank as i64;
    }
    let start = start.max(0) as usize;

    let mut end = shape_node
        .attrs
        .get("end")
        .map(|v| v.clone().into_i64())
        .unwrap_or(rank as i64);
    if end < 0 {
        end += rank as i64;
    }
    let end = (end as usize).min(rank);

    if start >= end {
        return Some(vec![]);
    }

    Some(static_shape[start..end].iter().map(|&d| d as i64).collect())
}

/// Check if a Slice node consumes a Shape node with static shape, and all slice
/// parameters (starts, ends, axes, steps) are constant. If so, compute the result.
fn extract_constant_shape_slice(
    slice: &RawNode,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
) -> Option<Vec<i64>> {
    // Slice needs at least 3 inputs: data, starts, ends
    if slice.inputs.len() < 3 {
        return None;
    }

    // Slice's data input must come from a Shape node with static shape
    let shape_idx = *producer.get(&slice.inputs[0].name)?;
    let shape_node = &nodes[shape_idx];
    if shape_node.node_type != NodeType::Shape {
        return None;
    }

    let shape_values = extract_full_static_shape(shape_node)?;
    let shape_len = shape_values.len() as i64;

    // Extract constant starts
    let starts = slice.inputs[1].value()?.to_i64_vec().ok()?;
    // Extract constant ends
    let ends = slice.inputs[2].value()?.to_i64_vec().ok()?;

    if starts.len() != ends.len() {
        return None;
    }

    // Extract optional axes (default: [0])
    let axes: Vec<i64> = if slice.inputs.len() > 3 {
        match slice.inputs[3].value() {
            Some(data) => data.to_i64_vec().ok()?,
            None => return None,
        }
    } else {
        (0..starts.len() as i64).collect()
    };

    // Extract optional steps (default: [1, 1, ...])
    let steps: Vec<i64> = if slice.inputs.len() > 4 {
        match slice.inputs[4].value() {
            Some(data) => data.to_i64_vec().ok()?,
            None => return None,
        }
    } else {
        vec![1; starts.len()]
    };

    if axes.len() != starts.len() || steps.len() != starts.len() {
        return None;
    }

    // Shape output is 1D, so the only valid axis is 0
    if axes.iter().any(|&a| a != 0) {
        return None;
    }

    // Apply the slice (1D case: single axis=0)
    // Use the first (and only) set of start/end/step values
    let mut start = starts[0];
    let mut end = ends[0];
    let step = steps[0];

    if step == 0 {
        return None;
    }

    // Normalize negative indices
    if start < 0 {
        start += shape_len;
    }
    if end < 0 {
        end += shape_len;
    }

    // Clamp to bounds
    start = start.clamp(0, shape_len);
    end = end.clamp(0, shape_len);

    // Compute the sliced result
    let result: Vec<i64> = if step > 0 {
        (start..end)
            .step_by(step as usize)
            .map(|i| shape_values[i as usize])
            .collect()
    } else {
        // Negative step: iterate from start down to end (exclusive)
        let mut vals = Vec::new();
        let mut i = start;
        while i > end {
            vals.push(shape_values[i as usize]);
            i += step; // step is negative
        }
        vals
    };

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, TensorType};
    use crate::simplify::tests::arg;
    use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};

    fn tensor_arg_with_shape(name: &str, shape: Vec<usize>) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: shape.len(),
                static_shape: Some(shape.into_iter().map(Some).collect()),
            }),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn shape_arg(name: &str, rank: usize) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Shape(rank),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn scalar_arg(name: &str, dtype: DType) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::ScalarNative(dtype),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn const_scalar_arg(name: &str, value: i64) -> Argument {
        Argument::from_const_i64(name, value)
    }

    fn const_i64_vec_arg(name: &str, values: &[i64]) -> Argument {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let data_ref =
            TensorDataRef::new(bytes::Bytes::from(bytes), vec![values.len()], DType::I64);
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
            ty: ArgType::Shape(values.len()),
            value_source: ValueSource::Constant,
            value_store: Some(value_store),
        }
    }

    fn raw_node(
        name: &str,
        node_type: NodeType,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        attrs: HashMap<String, AttributeValue>,
    ) -> RawNode {
        RawNode {
            node_type,
            name: name.to_string(),
            inputs,
            outputs,
            attrs,
        }
    }

    fn test_state() -> Rc<RefCell<GraphState>> {
        Rc::new(RefCell::new(GraphState::new(&[], &[], &[], &[])))
    }

    // --- Shape->Gather tests ---

    #[test]
    fn test_shape_gather_replaced_with_constant() {
        // tensor(shape=[2,3,4]) -> Shape -> Gather(idx=1) -> should become const 3
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4])],
                vec![shape_arg("shape_out", 3)],
                HashMap::new(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![shape_arg("shape_out", 3), const_scalar_arg("idx", 1)],
                vec![scalar_arg("dim_val", DType::I64)],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        let gather = result.iter().find(|n| n.name == "gather").unwrap();
        assert_eq!(gather.node_type, NodeType::Constant);

        // The input should be a constant with value 3
        let val = gather.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 3);
    }

    #[test]
    fn test_shape_with_start_attr() {
        // tensor(shape=[2,3,4,5]) -> Shape(start=1) -> Gather(idx=1) -> should be 4
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4, 5])],
                vec![shape_arg("shape_out", 3)],
                [("start".to_string(), AttributeValue::Int64(1))]
                    .into_iter()
                    .collect(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![shape_arg("shape_out", 3), const_scalar_arg("idx", 1)],
                vec![scalar_arg("dim_val", DType::I64)],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        let gather = result.iter().find(|n| n.name == "gather").unwrap();
        assert_eq!(gather.node_type, NodeType::Constant);
        let val = gather.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 4); // static_shape[1 + 1] = static_shape[2] = 4
    }

    #[test]
    fn test_gather_from_non_shape_not_replaced() {
        // Gather from a Relu output, not a Shape output
        let nodes = vec![
            raw_node(
                "relu",
                NodeType::Relu,
                vec![arg("input")],
                vec![arg("relu_out")],
                HashMap::new(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![arg("relu_out"), const_scalar_arg("idx", 0)],
                vec![arg("out")],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        assert_eq!(result[1].node_type, NodeType::Gather);
    }

    #[test]
    fn test_shape_without_static_shape_not_replaced() {
        // Shape input has no static_shape (dynamic)
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![arg("input")], // arg() creates tensor with static_shape: None
                vec![shape_arg("shape_out", 2)],
                HashMap::new(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![shape_arg("shape_out", 2), const_scalar_arg("idx", 0)],
                vec![arg("dim_val")],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        assert_eq!(result[1].node_type, NodeType::Gather);
    }

    // --- Shape elimination tests ---

    #[test]
    fn test_shape_not_replaced_without_gather_or_slice() {
        // Full Shape elimination is disabled because static_shape may not reflect
        // runtime dimensions for models with dynamic axes. Shape nodes without
        // downstream Gather/Slice consumers are left as-is.
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4])],
                vec![shape_arg("shape_out", 3)],
                HashMap::new(),
            ),
            raw_node(
                "consumer",
                NodeType::Add,
                vec![shape_arg("shape_out", 3), arg("other")],
                vec![arg("output")],
                HashMap::new(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        let shape = result.iter().find(|n| n.name == "shape").unwrap();
        assert_eq!(shape.node_type, NodeType::Shape); // NOT replaced
    }

    #[test]
    fn test_shape_dynamic_not_replaced() {
        let nodes = vec![raw_node(
            "shape",
            NodeType::Shape,
            vec![arg("input")], // no static_shape
            vec![shape_arg("shape_out", 2)],
            HashMap::new(),
        )];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        assert_eq!(result[0].node_type, NodeType::Shape);
    }

    // --- Shape->Slice tests ---

    #[test]
    fn test_shape_slice_replaced_with_constant() {
        // tensor(shape=[2,3,4,5]) -> Shape -> Slice(starts=[1], ends=[3]) -> [3, 4]
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4, 5])],
                vec![shape_arg("shape_out", 4)],
                HashMap::new(),
            ),
            raw_node(
                "slice",
                NodeType::Slice,
                vec![
                    shape_arg("shape_out", 4),
                    const_i64_vec_arg("starts", &[1]),
                    const_i64_vec_arg("ends", &[3]),
                ],
                vec![shape_arg("slice_out", 2)],
                HashMap::new(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        let slice = result.iter().find(|n| n.name == "slice").unwrap();
        assert_eq!(slice.node_type, NodeType::Constant);
        let vals = slice.inputs[0].value().unwrap().to_i64_vec().unwrap();
        assert_eq!(vals, vec![3, 4]);
    }

    #[test]
    fn test_shape_slice_with_step() {
        // tensor(shape=[10,20,30,40,50]) -> Shape -> Slice(starts=[0], ends=[5], axes=[0], steps=[2])
        // -> [10, 30, 50]
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![10, 20, 30, 40, 50])],
                vec![shape_arg("shape_out", 5)],
                HashMap::new(),
            ),
            raw_node(
                "slice",
                NodeType::Slice,
                vec![
                    shape_arg("shape_out", 5),
                    const_i64_vec_arg("starts", &[0]),
                    const_i64_vec_arg("ends", &[5]),
                    const_i64_vec_arg("axes", &[0]),
                    const_i64_vec_arg("steps", &[2]),
                ],
                vec![shape_arg("slice_out", 3)],
                HashMap::new(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        let slice = result.iter().find(|n| n.name == "slice").unwrap();
        assert_eq!(slice.node_type, NodeType::Constant);
        let vals = slice.inputs[0].value().unwrap().to_i64_vec().unwrap();
        assert_eq!(vals, vec![10, 30, 50]);
    }

    #[test]
    fn test_shape_slice_negative_index() {
        // tensor(shape=[2,3,4]) -> Shape -> Slice(starts=[-2], ends=[3]) -> [3, 4]
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4])],
                vec![shape_arg("shape_out", 3)],
                HashMap::new(),
            ),
            raw_node(
                "slice",
                NodeType::Slice,
                vec![
                    shape_arg("shape_out", 3),
                    const_i64_vec_arg("starts", &[-2]),
                    const_i64_vec_arg("ends", &[3]),
                ],
                vec![shape_arg("slice_out", 2)],
                HashMap::new(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        let slice = result.iter().find(|n| n.name == "slice").unwrap();
        assert_eq!(slice.node_type, NodeType::Constant);
        let vals = slice.inputs[0].value().unwrap().to_i64_vec().unwrap();
        assert_eq!(vals, vec![3, 4]);
    }

    #[test]
    fn test_slice_from_non_shape_not_replaced() {
        // Slice from Relu output, not Shape
        let nodes = vec![
            raw_node(
                "relu",
                NodeType::Relu,
                vec![arg("input")],
                vec![arg("relu_out")],
                HashMap::new(),
            ),
            raw_node(
                "slice",
                NodeType::Slice,
                vec![
                    arg("relu_out"),
                    const_i64_vec_arg("starts", &[0]),
                    const_i64_vec_arg("ends", &[2]),
                ],
                vec![arg("slice_out")],
                HashMap::new(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        assert_eq!(result[1].node_type, NodeType::Slice);
    }

    #[test]
    fn test_shape_slice_dynamic_starts_not_replaced() {
        // Slice with dynamic (non-constant) starts should not be replaced
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4])],
                vec![shape_arg("shape_out", 3)],
                HashMap::new(),
            ),
            raw_node(
                "slice",
                NodeType::Slice,
                vec![
                    shape_arg("shape_out", 3),
                    arg("dynamic_starts"), // no constant value
                    const_i64_vec_arg("ends", &[3]),
                ],
                vec![shape_arg("slice_out", 2)],
                HashMap::new(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);
        assert_eq!(result[1].node_type, NodeType::Slice);
    }

    #[test]
    fn test_value_store_propagated_to_downstream_inputs() {
        // Shape(x=[2,3,4]) -> Gather(idx=1) -> downstream Add
        // After folding, the Add's input referencing the Gather output should have value_store
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4])],
                vec![shape_arg("shape_out", 3)],
                HashMap::new(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![shape_arg("shape_out", 3), const_scalar_arg("idx", 1)],
                vec![scalar_arg("dim_val", DType::I64)],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
            raw_node(
                "add",
                NodeType::Add,
                vec![scalar_arg("dim_val", DType::I64), arg("other")],
                vec![arg("add_out")],
                HashMap::new(),
            ),
        ];

        let state = test_state();
        let result = simplify_constant_shape(nodes, &mut [], &state);

        // The Add's first input should now have value_store and be readable
        let add_node = result.iter().find(|n| n.name == "add").unwrap();
        assert_eq!(add_node.inputs[0].value_source, ValueSource::Constant);
        let val = add_node.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 3);
    }

    #[test]
    fn test_value_store_propagated_to_graph_outputs() {
        // Shape(x=[2,3,4]) -> Gather(idx=0) -> graph output
        // After folding, the graph output should have value_store
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4])],
                vec![shape_arg("shape_out", 3)],
                HashMap::new(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![shape_arg("shape_out", 3), const_scalar_arg("idx", 0)],
                vec![scalar_arg("dim_val", DType::I64)],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let mut graph_outputs = vec![scalar_arg("dim_val", DType::I64)];
        let state = test_state();
        simplify_constant_shape(nodes, &mut graph_outputs, &state);

        assert_eq!(graph_outputs[0].value_source, ValueSource::Constant);
        let val = graph_outputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 2);
    }
}
