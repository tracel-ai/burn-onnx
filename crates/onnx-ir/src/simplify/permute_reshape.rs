use std::collections::HashMap;

use crate::TensorDataExt;
use crate::ir::{AttributeValue, NodeType, RawNode};

/// Detect Shape+Gather+Unsqueeze+Concat+Reshape chains that implement a dimension
/// permutation (e.g. from PyTorch's `tensor.permute()`) and replace the Reshape with
/// a Transpose node. The orphaned Shape/Gather/Unsqueeze/Concat nodes are cleaned up
/// by dead node elimination.
///
/// Pattern:
///   input -> Shape -> Gather(idx=i) -> Unsqueeze -> ┐
///                  -> Gather(idx=j) -> Unsqueeze -> ├─ Concat -> Reshape(input) -> output
///                  -> Gather(idx=k) -> Unsqueeze -> ┘
pub(crate) fn simplify_permute_reshape(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    // Build output_name -> node index map for fast lookup
    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.name.clone(), i);
        }
    }

    // Collect replacements: (reshape_node_index, perm)
    let mut replacements: Vec<(usize, Vec<i64>)> = Vec::new();

    for (ri, node) in nodes.iter().enumerate() {
        if node.node_type != NodeType::Reshape || node.inputs.len() < 2 {
            continue;
        }
        if let Some(perm) = extract_permute_pattern(node, &nodes, &producer) {
            replacements.push((ri, perm));
        }
    }

    // Apply replacements
    for (ri, perm) in &replacements {
        let reshape = &nodes[*ri];
        log::info!(
            "Simplification: replacing permute-reshape '{}' with Transpose(perm={:?})",
            reshape.name,
            perm,
        );

        // Build Transpose node reusing the Reshape's data input and output
        let mut attrs = HashMap::new();
        attrs.insert("perm".to_string(), AttributeValue::Int64s(perm.clone()));

        nodes[*ri] = RawNode {
            node_type: NodeType::Transpose,
            name: nodes[*ri].name.clone(),
            inputs: vec![nodes[*ri].inputs[0].clone()],
            outputs: nodes[*ri].outputs.clone(),
            attrs,
        };
    }

    nodes
}

/// Try to extract a permutation from the Reshape's shape-input chain.
/// Returns `Some(perm)` if the pattern matches, `None` otherwise.
fn extract_permute_pattern(
    reshape: &RawNode,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
) -> Option<Vec<i64>> {
    // The shape input is input[1] for Reshape
    let shape_input_name = &reshape.inputs[1].name;
    let concat_idx = *producer.get(shape_input_name)?;
    let concat = &nodes[concat_idx];

    if concat.node_type != NodeType::Concat {
        return None;
    }

    // Each Concat input should come from an Unsqueeze -> Gather -> Shape chain,
    // all from the same Shape node, and the Reshape's data input should match
    // the Shape's data input.
    let mut perm: Vec<i64> = Vec::with_capacity(concat.inputs.len());
    let mut shape_data_input: Option<String> = None;

    for concat_in in &concat.inputs {
        let unsqueeze_idx = *producer.get(&concat_in.name)?;
        let unsqueeze = &nodes[unsqueeze_idx];
        if unsqueeze.node_type != NodeType::Unsqueeze {
            return None;
        }

        let gather_idx = *producer.get(&unsqueeze.inputs[0].name)?;
        let gather = &nodes[gather_idx];
        if gather.node_type != NodeType::Gather {
            return None;
        }

        // Gather's axis must be 0
        let axis = gather
            .attrs
            .get("axis")
            .map(|v| v.clone().into_i64())
            .unwrap_or(0);
        if axis != 0 {
            return None;
        }

        // Gather's index (input[1]) must be a constant scalar
        let index_arg = gather.inputs.get(1)?;
        let mut index_val = index_arg.value()?.scalar_i64().ok()?;
        // Normalize negative indices (ONNX Gather supports them)
        if index_val < 0 {
            index_val += concat.inputs.len() as i64;
        }
        perm.push(index_val);

        // Gather's data input (input[0]) must come from a Shape node
        let shape_idx = *producer.get(&gather.inputs[0].name)?;
        let shape_node = &nodes[shape_idx];
        if shape_node.node_type != NodeType::Shape {
            return None;
        }

        // All Shape nodes must reference the same data input
        let this_data_input = &shape_node.inputs[0].name;
        match &shape_data_input {
            None => shape_data_input = Some(this_data_input.clone()),
            Some(prev) if prev != this_data_input => return None,
            _ => {}
        }
    }

    // The Reshape's data input (input[0]) must be the same tensor that Shape reads
    let reshape_data_input = &reshape.inputs[0].name;
    if shape_data_input.as_ref() != Some(reshape_data_input) {
        return None;
    }

    // Verify it's actually a permutation (contains each index exactly once)
    let rank = perm.len();
    let mut seen = vec![false; rank];
    for &p in &perm {
        if p < 0 || p as usize >= rank || seen[p as usize] {
            return None;
        }
        seen[p as usize] = true;
    }

    Some(perm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, ValueSource};
    use crate::simplify::tests::{arg, node};
    use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};

    /// Create a value store with scalar i64 constants.
    /// Maps name -> i64 value.
    fn make_value_store(constants: &[(&str, i64)]) -> ValueStore {
        let mut store = TensorStore::new();
        let mut constant_map = std::collections::HashMap::new();
        for &(name, val) in constants {
            let bytes = val.to_ne_bytes().to_vec();
            let data_ref = TensorDataRef::new(bytes::Bytes::from(bytes), vec![1], DType::I64);
            let id = store.store(data_ref);
            constant_map.insert(name.to_string(), id);
        }
        ValueStore::new(std::rc::Rc::new(store), std::rc::Rc::new(constant_map))
    }

    fn const_scalar_arg(name: &str, store: &ValueStore) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Scalar(DType::I64),
            value_source: ValueSource::Constant,
            value_store: Some(store.clone()),
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

    fn node_with_attrs(
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

    /// Build the full permute pattern for a 3D tensor with perm [0, 2, 1]:
    ///   input(rank=3) -> Shape -> Gather(0) -> Unsqueeze -> ┐
    ///                         -> Gather(2) -> Unsqueeze -> ├─ Concat -> Reshape -> output
    ///                         -> Gather(1) -> Unsqueeze -> ┘
    fn build_permute_pattern(perm: &[i64]) -> Vec<RawNode> {
        let rank = perm.len();
        let store = make_value_store(
            &perm
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let name: &'static str = Box::leak(format!("idx_{}", i).into_boxed_str());
                    (name, p)
                })
                .collect::<Vec<_>>(),
        );

        let input = arg("input");
        let mut nodes = Vec::new();

        // Shape node
        nodes.push(node_with_attrs(
            "shape",
            NodeType::Shape,
            vec![input.clone()],
            vec![shape_arg("shape_out", rank)],
            HashMap::new(),
        ));

        let mut concat_inputs = Vec::new();

        for (i, _) in perm.iter().enumerate() {
            let gather_out = format!("gather_{}_out", i);
            let unsqueeze_out = format!("unsqueeze_{}_out", i);
            let idx_name = format!("idx_{}", i);

            // Gather node
            nodes.push(node_with_attrs(
                &format!("gather_{}", i),
                NodeType::Gather,
                vec![
                    shape_arg("shape_out", rank),
                    const_scalar_arg(&idx_name, &store),
                ],
                vec![arg(&gather_out)],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ));

            // Unsqueeze node
            nodes.push(node_with_attrs(
                &format!("unsqueeze_{}", i),
                NodeType::Unsqueeze,
                vec![arg(&gather_out)],
                vec![shape_arg(&unsqueeze_out, 1)],
                HashMap::new(),
            ));

            concat_inputs.push(shape_arg(&unsqueeze_out, 1));
        }

        // Concat node
        nodes.push(node_with_attrs(
            "concat",
            NodeType::Concat,
            concat_inputs,
            vec![shape_arg("new_shape", rank)],
            [("axis".to_string(), AttributeValue::Int64(0))]
                .into_iter()
                .collect(),
        ));

        // Reshape node
        nodes.push(node_with_attrs(
            "reshape",
            NodeType::Reshape,
            vec![input, shape_arg("new_shape", rank)],
            vec![arg("output")],
            HashMap::new(),
        ));

        nodes
    }

    #[test]
    fn test_permute_pattern_detected() {
        let nodes = build_permute_pattern(&[0, 2, 1]);
        let result = simplify_permute_reshape(nodes);

        // Reshape should be replaced with Transpose
        let reshape_node = result.iter().find(|n| n.name == "reshape").unwrap();
        assert_eq!(reshape_node.node_type, NodeType::Transpose);
        let perm = reshape_node.attrs.get("perm").unwrap().clone().into_i64s();
        assert_eq!(perm, vec![0, 2, 1]);
        // Should have 1 input (the data tensor), not 2
        assert_eq!(reshape_node.inputs.len(), 1);
        assert_eq!(reshape_node.inputs[0].name, "input");
    }

    #[test]
    fn test_permute_pattern_4d() {
        let nodes = build_permute_pattern(&[0, 3, 1, 2]);
        let result = simplify_permute_reshape(nodes);

        let node = result.iter().find(|n| n.name == "reshape").unwrap();
        assert_eq!(node.node_type, NodeType::Transpose);
        let perm = node.attrs.get("perm").unwrap().clone().into_i64s();
        assert_eq!(perm, vec![0, 3, 1, 2]);
    }

    #[test]
    fn test_permute_pattern_identity_still_matches() {
        // Identity perm [0,1,2] still matches (dead node elim or noop handles it)
        let nodes = build_permute_pattern(&[0, 1, 2]);
        let result = simplify_permute_reshape(nodes);

        let node = result.iter().find(|n| n.name == "reshape").unwrap();
        assert_eq!(node.node_type, NodeType::Transpose);
    }

    #[test]
    fn test_non_permute_reshape_not_affected() {
        // A plain Reshape with no pattern chain should be untouched
        let nodes = vec![node(
            "reshape",
            NodeType::Reshape,
            &["input", "shape"],
            &["output"],
        )];
        let result = simplify_permute_reshape(nodes);
        assert_eq!(result[0].node_type, NodeType::Reshape);
    }
}
