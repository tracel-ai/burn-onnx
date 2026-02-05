//! Graph simplification passes
//!
//! Runs optimization passes on the IR graph after post-processing but before finalization.
//! Each pass is a function that takes and returns `(nodes, inputs, outputs)`.
//!
//! ## Current passes (in execution order per iteration)
//!
//! 1. **Permute-reshape detection** - Shape+Gather+Unsqueeze+Concat+Reshape -> Transpose
//! 2. **Constant shape propagation** - Shape->Gather and Shape->Slice elimination
//! 3. **Idempotent op elimination** - f(f(x)) -> f(x) for Relu, Ceil, Floor, etc.
//! 4. **Identity element elimination** - x+0, x*1, x/1, x**1 -> x
//! 5. **Common subexpression elimination** - merge duplicate nodes
//! 6. **Dead node elimination** - remove unreferenced nodes (cascading)
//!
//! All passes run in a fixed-point loop until the graph stabilizes.
//!
//! ## Future work: Constant folding
//!
//! TODO(#70): Add constant folding pass that evaluates nodes with all-constant inputs at compile
//! time. This would replace arbitrary constant expressions (e.g., Const(2) + Const(3) -> Const(5))
//! beyond the shape-specific patterns already handled.

mod constant_shape;
mod dead_nodes;
mod idempotent;
mod identity_element;
mod permute_reshape;
mod redundant_nodes;

use std::{cell::RefCell, rc::Rc};

use crate::{
    graph_state::GraphState,
    ir::{Argument, RawNode},
};

use constant_shape::simplify_constant_shape;
use dead_nodes::eliminate_dead_nodes;
use idempotent::eliminate_idempotent_ops;
use identity_element::eliminate_identity_elements;
use permute_reshape::simplify_permute_reshape;
use redundant_nodes::eliminate_redundant_nodes;

/// Maximum number of fixed-point iterations to prevent runaway loops.
const MAX_ITERATIONS: usize = 10;

/// Run all simplification passes on the graph.
///
/// Applies passes in a fixed-point loop: repeats until the graph stops changing
/// or `MAX_ITERATIONS` is reached. Each iteration runs all passes in order,
/// since one pass may create opportunities for another.
pub(crate) fn simplify_graph(
    mut nodes: Vec<RawNode>,
    inputs: Vec<Argument>,
    mut outputs: Vec<Argument>,
    _state: &Rc<RefCell<GraphState>>,
) -> (Vec<RawNode>, Vec<Argument>, Vec<Argument>) {
    for iteration in 0..MAX_ITERATIONS {
        let node_count_before = nodes.len();

        // Structural pattern detection (must run before constant folding, which
        // replaces Gather/Slice nodes with Constants and destroys the patterns)
        nodes = simplify_permute_reshape(nodes);

        // Constant propagation (may eliminate Shape->Gather chains)
        nodes = simplify_constant_shape(nodes, &mut outputs, _state);

        // Idempotent op elimination: f(f(x)) -> f(x)
        nodes = eliminate_idempotent_ops(nodes);

        // Identity element elimination: x + 0 -> x, x * 1 -> x, etc.
        nodes = eliminate_identity_elements(nodes);

        // Common subexpression elimination (rewrites inputs, creates dead nodes)
        nodes = eliminate_redundant_nodes(nodes);

        // Dead node elimination (cleans up nodes orphaned by pattern passes)
        nodes = eliminate_dead_nodes(nodes, &outputs);

        let removed = node_count_before - nodes.len();
        if removed == 0 {
            log::debug!(
                "Simplification: converged after {} iteration(s)",
                iteration + 1
            );
            break;
        }

        log::info!(
            "Simplification: iteration {} removed {} node(s)",
            iteration + 1,
            removed
        );
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
