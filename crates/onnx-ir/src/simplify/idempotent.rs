use std::collections::HashMap;

use crate::ir::{NodeType, RawNode};

/// Ops where applying the same op twice is equivalent to applying it once: `f(f(x)) == f(x)`.
const IDEMPOTENT_OPS: &[NodeType] = &[
    NodeType::Relu,
    NodeType::Ceil,
    NodeType::Floor,
    NodeType::Round,
    NodeType::Sign,
    NodeType::Abs,
];

/// Eliminate consecutive applications of idempotent unary ops.
///
/// For ops like Relu, Ceil, Floor, etc., `f(f(x))` produces the same result as `f(x)`.
/// When we detect a node whose single input is the output of the same op type, we rewire
/// the second node's consumers to use the first node's output. Dead node elimination
/// cleans up the now-unused second node.
pub(crate) fn eliminate_idempotent_ops(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    // Build a map: output_name -> (node_index, node_type)
    let mut output_map: HashMap<&str, (usize, &NodeType)> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            output_map.insert(&out.name, (i, &node.node_type));
        }
    }

    // Find idempotent chains: if node B has same op type as node A and B's input is A's output,
    // rewire B's consumers to use A's output directly (making B dead).
    let mut rename: HashMap<String, String> = HashMap::new();

    for node in &nodes {
        if !IDEMPOTENT_OPS.contains(&node.node_type) {
            continue;
        }
        if node.inputs.len() != 1 || node.outputs.len() != 1 {
            continue;
        }

        let input_name = &node.inputs[0].name;
        if let Some(&(_, producer_type)) = output_map.get(input_name.as_str())
            && *producer_type == node.node_type
        {
            // f(f(x)) -> f(x): rewire this node's output to its input (the first f's output)
            log::debug!(
                "Idempotent elimination: {:?} '{}' (output '{}' -> input '{}')",
                node.node_type,
                node.name,
                node.outputs[0].name,
                input_name,
            );
            rename.insert(node.outputs[0].name.clone(), input_name.clone());
        }
    }

    if rename.is_empty() {
        return nodes;
    }

    log::info!(
        "Idempotent elimination: found {} redundant op(s)",
        rename.len()
    );

    // Rewrite inputs referencing the redundant outputs
    for node in &mut nodes {
        for input in &mut node.inputs {
            if let Some(new_name) = rename.get(&input.name) {
                input.name = new_name.clone();
            }
        }
    }

    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplify::tests::node;

    #[test]
    fn test_relu_relu_eliminated() {
        let nodes = vec![
            node("relu1", NodeType::Relu, &["input"], &["r1_out"]),
            node("relu2", NodeType::Relu, &["r1_out"], &["r2_out"]),
            node("add", NodeType::Add, &["r2_out", "other"], &["output"]),
        ];

        let result = eliminate_idempotent_ops(nodes);

        // relu2's output should be rewritten to relu1's output in add's inputs
        let add = result.iter().find(|n| n.name == "add").unwrap();
        assert_eq!(add.inputs[0].name, "r1_out");
    }

    #[test]
    fn test_different_ops_not_eliminated() {
        let nodes = vec![
            node("relu1", NodeType::Relu, &["input"], &["r1_out"]),
            node("ceil1", NodeType::Ceil, &["r1_out"], &["c1_out"]),
        ];

        let result = eliminate_idempotent_ops(nodes);

        let ceil = result.iter().find(|n| n.name == "ceil1").unwrap();
        assert_eq!(ceil.inputs[0].name, "r1_out");
    }

    #[test]
    fn test_triple_chain_partially_eliminated() {
        // relu(relu(relu(x))) -> after one pass, becomes relu(relu(x)) with dead relu3
        // (the second relu(relu) would need another pass or dead node + re-run)
        let nodes = vec![
            node("relu1", NodeType::Relu, &["input"], &["r1_out"]),
            node("relu2", NodeType::Relu, &["r1_out"], &["r2_out"]),
            node("relu3", NodeType::Relu, &["r2_out"], &["r3_out"]),
            node("out", NodeType::Add, &["r3_out", "other"], &["output"]),
        ];

        let result = eliminate_idempotent_ops(nodes);

        // relu2 -> r1_out, relu3 -> r2_out (which was its input)
        // But relu3's input r2_out gets rewritten too... let's check:
        // relu3 reads r2_out, producer is relu2 (same type) -> rename r3_out -> r2_out
        // relu2 reads r1_out, producer is relu1 (same type) -> rename r2_out -> r1_out
        // Then rewrite: out's input r3_out -> r2_out, and r2_out -> r1_out
        // But rename only does one level. So out gets r2_out, and relu3 keeps r2_out.
        // After dead node elimination + re-run it would collapse fully.
        let out = result.iter().find(|n| n.name == "out").unwrap();
        assert_eq!(out.inputs[0].name, "r2_out");
    }

    #[test]
    fn test_non_idempotent_op_not_eliminated() {
        // Sigmoid is not idempotent
        let nodes = vec![
            node("sig1", NodeType::Sigmoid, &["input"], &["s1_out"]),
            node("sig2", NodeType::Sigmoid, &["s1_out"], &["s2_out"]),
        ];

        let result = eliminate_idempotent_ops(nodes);

        let sig2 = result.iter().find(|n| n.name == "sig2").unwrap();
        assert_eq!(sig2.inputs[0].name, "s1_out");
    }
}
