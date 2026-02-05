use std::collections::HashSet;

use crate::ir::{Argument, RawNode};

/// Remove nodes whose outputs are not consumed by any downstream node or graph output.
///
/// Walks the node list and collects all output names that are "live" (referenced as an
/// input by a later node or listed as a graph output). Nodes with no live outputs are removed.
/// Repeats until stable, since removing a node may make its predecessors dead.
pub(crate) fn eliminate_dead_nodes(
    mut nodes: Vec<RawNode>,
    graph_outputs: &[Argument],
) -> Vec<RawNode> {
    loop {
        // Collect all names that are consumed: graph outputs + all node inputs
        let mut live_names: HashSet<String> = HashSet::new();
        for output in graph_outputs {
            live_names.insert(output.name.clone());
        }
        for node in &nodes {
            for input in &node.inputs {
                live_names.insert(input.name.clone());
            }
        }

        // A node is dead if none of its outputs appear in live_names
        let before = nodes.len();
        nodes.retain(|node| {
            let alive = node
                .outputs
                .iter()
                .any(|out| live_names.contains(&out.name));
            if !alive {
                log::debug!(
                    "Dead node elimination: {:?} '{}'",
                    node.node_type,
                    node.name,
                );
            }
            alive
        });

        if nodes.len() == before {
            break;
        }
        // Removing nodes may expose new dead nodes, so loop again
    }

    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::simplify::tests::{arg, node};

    #[test]
    fn test_no_dead_nodes() {
        // A -> B -> output
        let nodes = vec![
            node("a", NodeType::Relu, &["input"], &["a_out"]),
            node("b", NodeType::Relu, &["a_out"], &["output"]),
        ];
        let outputs = vec![arg("output")];

        let result = eliminate_dead_nodes(nodes, &outputs);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_single_dead_node() {
        // A -> output, B is unused
        let nodes = vec![
            node("a", NodeType::Relu, &["input"], &["output"]),
            node("b", NodeType::Relu, &["input"], &["unused"]),
        ];
        let outputs = vec![arg("output")];

        let result = eliminate_dead_nodes(nodes, &outputs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "a");
    }

    #[test]
    fn test_cascading_dead_nodes() {
        // A feeds B, B is dead, so A becomes dead too.
        // C -> output is the only live path.
        let nodes = vec![
            node("a", NodeType::Relu, &["input"], &["a_out"]),
            node("b", NodeType::Relu, &["a_out"], &["b_out"]),
            node("c", NodeType::Relu, &["input"], &["output"]),
        ];
        let outputs = vec![arg("output")];

        let result = eliminate_dead_nodes(nodes, &outputs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "c");
    }

    #[test]
    fn test_node_with_multiple_outputs_partially_used() {
        // Node A has two outputs, only one is consumed
        let nodes = vec![
            node("a", NodeType::Relu, &["input"], &["a_out1", "a_out2"]),
            node("b", NodeType::Relu, &["a_out1"], &["output"]),
        ];
        let outputs = vec![arg("output")];

        let result = eliminate_dead_nodes(nodes, &outputs);
        // A stays because a_out1 is used by B
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_all_nodes_dead() {
        let nodes = vec![
            node("a", NodeType::Relu, &["input"], &["a_out"]),
            node("b", NodeType::Relu, &["a_out"], &["b_out"]),
        ];
        // Graph output references something these nodes don't produce
        let outputs = vec![arg("external")];

        let result = eliminate_dead_nodes(nodes, &outputs);
        assert_eq!(result.len(), 0);
    }
}
