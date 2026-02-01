//! Graph simplification passes
//!
//! Runs optimization passes on the IR graph after post-processing but before finalization.
//! Each pass is a function that takes and returns `(nodes, inputs, outputs)`.

mod constant_shape;
mod dead_nodes;
mod permute_reshape;
mod redundant_nodes;

use std::{cell::RefCell, rc::Rc};

use crate::{
    graph_state::GraphState,
    ir::{Argument, RawNode},
};

use constant_shape::simplify_constant_shape;
use dead_nodes::eliminate_dead_nodes;
use permute_reshape::simplify_permute_reshape;
use redundant_nodes::eliminate_redundant_nodes;

/// Run all simplification passes on the graph.
///
/// Returns the (possibly modified) nodes, inputs, and outputs.
pub(crate) fn simplify_graph(
    nodes: Vec<RawNode>,
    inputs: Vec<Argument>,
    outputs: Vec<Argument>,
    _state: &Rc<RefCell<GraphState>>,
) -> (Vec<RawNode>, Vec<Argument>, Vec<Argument>) {
    // Constant propagation (may eliminate Shape->Gather chains)
    let nodes = simplify_constant_shape(nodes);

    // Pattern-based simplifications (may create dead nodes)
    let nodes = simplify_permute_reshape(nodes);

    // Common subexpression elimination (rewrites inputs, creates dead nodes)
    let nodes = eliminate_redundant_nodes(nodes);

    // Dead node elimination (cleans up nodes orphaned by pattern passes)
    let node_count_before = nodes.len();
    let nodes = eliminate_dead_nodes(nodes, &outputs);
    let removed = node_count_before - nodes.len();
    if removed > 0 {
        log::info!("Simplification: removed {} dead node(s)", removed);
    } else {
        log::debug!("Simplification: no dead nodes found");
    }

    (nodes, inputs, outputs)
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::ir::{ArgType, Argument, DType, NodeType, RawNode, TensorType, ValueSource};

    pub fn arg(name: &str) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 2,
                static_shape: None,
            }),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    pub fn node(name: &str, node_type: NodeType, inputs: &[&str], outputs: &[&str]) -> RawNode {
        RawNode {
            node_type,
            name: name.to_string(),
            inputs: inputs.iter().map(|n| arg(n)).collect(),
            outputs: outputs.iter().map(|n| arg(n)).collect(),
            attrs: Default::default(),
        }
    }
}
