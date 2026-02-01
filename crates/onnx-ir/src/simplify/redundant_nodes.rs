use std::collections::HashMap;

use crate::ir::{AttributeValue, RawNode};

/// Eliminate redundant nodes (common subexpression elimination).
///
/// Two nodes are considered redundant if they have the same op type, the same input
/// names (in order), and the same attributes. When a duplicate is found, all references
/// to its outputs are rewritten to point to the original node's outputs. The duplicate
/// becomes dead and is cleaned up by dead node elimination.
pub(crate) fn eliminate_redundant_nodes(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    // key -> index of the first node with that key
    let mut seen: HashMap<String, usize> = HashMap::new();
    // duplicate output name -> original output name
    let mut rename: HashMap<String, String> = HashMap::new();

    for i in 0..nodes.len() {
        let key = cse_key(&nodes[i]);
        if let Some(&original_idx) = seen.get(&key) {
            // Map each output of the duplicate to the corresponding output of the original
            let original = &nodes[original_idx];
            let duplicate = &nodes[i];
            if original.outputs.len() == duplicate.outputs.len() {
                for (orig_out, dup_out) in
                    original.outputs.iter().zip(duplicate.outputs.iter())
                {
                    log::debug!(
                        "CSE: '{}' output '{}' -> '{}'",
                        duplicate.name,
                        dup_out.name,
                        orig_out.name,
                    );
                    rename.insert(dup_out.name.clone(), orig_out.name.clone());
                }
            }
        } else {
            seen.insert(key, i);
        }
    }

    if rename.is_empty() {
        return nodes;
    }

    log::info!(
        "CSE: found {} redundant output(s) to rewrite",
        rename.len()
    );

    // Rewrite all input names that reference a duplicate output
    for node in &mut nodes {
        for input in &mut node.inputs {
            if let Some(new_name) = rename.get(&input.name) {
                input.name = new_name.clone();
            }
        }
    }

    nodes
}

/// Build a CSE key for a node based on its type, input names, and attributes.
///
/// Nodes with graph-valued attributes get a unique key (not eligible for CSE).
fn cse_key(node: &RawNode) -> String {
    // Skip nodes with graph attributes (too complex to compare)
    for value in node.attrs.values() {
        if matches!(
            value,
            AttributeValue::DeferredGraph(_)
                | AttributeValue::DeferredGraphs(_)
                | AttributeValue::Graph(_)
        ) {
            // Return a unique key so this node never matches another
            return format!("__unique__:{}", node.name);
        }
    }

    let input_names: Vec<&str> = node.inputs.iter().map(|i| i.name.as_str()).collect();

    // Sort attributes by key for deterministic comparison
    let mut attrs: Vec<(&String, &AttributeValue)> = node.attrs.iter().collect();
    attrs.sort_by_key(|(k, _)| *k);

    format!("{:?}|{:?}|{:?}", node.node_type, input_names, attrs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, NodeType};
    use crate::simplify::tests::node;

    #[test]
    fn test_duplicate_nodes_merged() {
        // Two identical Relu nodes with same input, consumer uses second
        let nodes = vec![
            node("relu_1", NodeType::Relu, &["input"], &["r1_out"]),
            node("relu_2", NodeType::Relu, &["input"], &["r2_out"]),
            node("add", NodeType::Add, &["r1_out", "r2_out"], &["output"]),
        ];

        let result = eliminate_redundant_nodes(nodes);

        // relu_2's output should be rewritten to relu_1's output
        let add = result.iter().find(|n| n.name == "add").unwrap();
        assert_eq!(add.inputs[0].name, "r1_out");
        assert_eq!(add.inputs[1].name, "r1_out"); // was r2_out, now r1_out
    }

    #[test]
    fn test_different_inputs_not_merged() {
        let nodes = vec![
            node("relu_1", NodeType::Relu, &["input_a"], &["r1_out"]),
            node("relu_2", NodeType::Relu, &["input_b"], &["r2_out"]),
        ];

        let result = eliminate_redundant_nodes(nodes);
        // Both should remain unchanged
        assert_eq!(result[0].inputs[0].name, "input_a");
        assert_eq!(result[1].inputs[0].name, "input_b");
    }

    #[test]
    fn test_different_attrs_not_merged() {
        let mut n1 = node("gather_1", NodeType::Gather, &["data", "idx"], &["g1_out"]);
        n1.attrs
            .insert("axis".to_string(), AttributeValue::Int64(0));

        let mut n2 = node("gather_2", NodeType::Gather, &["data", "idx"], &["g2_out"]);
        n2.attrs
            .insert("axis".to_string(), AttributeValue::Int64(1));

        let nodes = vec![n1, n2];
        let result = eliminate_redundant_nodes(nodes);

        // Both kept, no rewriting
        assert_eq!(result[0].outputs[0].name, "g1_out");
        assert_eq!(result[1].outputs[0].name, "g2_out");
    }

    #[test]
    fn test_diamond_pattern_cse() {
        // A -> B and A -> C, where B == C (same op, same input)
        // Consumer uses both B and C outputs
        let nodes = vec![
            node("a", NodeType::Relu, &["input"], &["a_out"]),
            node("b", NodeType::Sigmoid, &["a_out"], &["b_out"]),
            node("c", NodeType::Sigmoid, &["a_out"], &["c_out"]),
            node("add", NodeType::Add, &["b_out", "c_out"], &["output"]),
        ];

        let result = eliminate_redundant_nodes(nodes);

        let add = result.iter().find(|n| n.name == "add").unwrap();
        // c_out should be rewritten to b_out
        assert_eq!(add.inputs[0].name, "b_out");
        assert_eq!(add.inputs[1].name, "b_out");
    }
}
