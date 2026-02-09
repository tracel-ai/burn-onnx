/// Tests for simplification passes.
///
/// Verifies that graph simplification correctly transforms patterns
/// into more efficient node representations.
mod test_utils;

use test_utils::*;

/// Pre-scaled SDPA pattern (as seen in RF-DETR) should be coalesced into a single
/// Attention node when simplification is enabled:
///
/// ```text
/// Q -> Transpose([0,2,1,3]) -> Mul(sqrt_scale) -\
///                                                 MatMul -> Softmax -> MatMul(scores, V)
/// K -> Transpose([0,2,3,1]) -> Mul(sqrt_scale) -/
/// ```
#[test]
fn test_prescaled_sdpa_coalesced() {
    let graph = load_onnx_simplified("prescaled_sdpa.onnx");

    // Find the Attention node
    let attention = graph
        .nodes
        .iter()
        .find(|n| matches!(n, onnx_ir::ir::Node::Attention { .. }));
    assert!(
        attention.is_some(),
        "Pre-scaled SDPA pattern should be coalesced into an Attention node"
    );

    // Verify scale is None (dynamic sqrt_scale computes to the default 1/sqrt(head_dim))
    if let Some(onnx_ir::ir::Node::Attention(node)) = attention {
        assert!(
            node.config.scale.is_none(),
            "Dynamic pre-scale should result in default scale (None)"
        );
    }

    // The pattern's MatMul, Softmax, Mul(scale) nodes should be eliminated
    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Softmax { .. })),
        "Softmax should be absorbed into Attention node"
    );
}

/// Without simplification, the pre-scaled SDPA pattern should remain decomposed.
#[test]
fn test_prescaled_sdpa_not_coalesced_without_simplify() {
    let graph = load_onnx("prescaled_sdpa.onnx");

    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Attention { .. })),
        "Attention node should not appear without simplification"
    );

    assert!(
        has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Softmax { .. })),
        "Softmax should remain without simplification"
    );
}

// --- Constant folding tests ---

/// Cascading fold: Shape->Gather produces constants, then Mul(const, const) folds.
///
/// x: [2, 3, 4]
/// Shape(x)[1]=3, Shape(x)[2]=4 => constant_shape folds the Gathers
/// Mul(3, 4)=12 => constant_fold folds the Mul
///
/// With simplification: only Constant nodes should remain (no Mul, no Shape, no Gather).
#[test]
fn test_constant_fold_cascade() {
    let graph = load_onnx_simplified("constant_fold_cascade.onnx");

    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Mul { .. })),
        "Mul should be folded into a constant"
    );
    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Gather { .. })),
        "Gather should be folded into a constant"
    );
    assert_eq!(
        count_operation_nodes(&graph),
        0,
        "only constants should remain"
    );
}

/// Without simplification, the Mul and Gathers remain as operations.
#[test]
fn test_constant_fold_cascade_without_simplify() {
    let graph = load_onnx("constant_fold_cascade.onnx");

    assert!(
        has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Mul { .. })),
        "Mul should remain without simplification"
    );
    assert!(
        has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Gather { .. })),
        "Gather should remain without simplification"
    );
}

/// Chained arithmetic on initializer constants: Add(2,3)=5, Sub(5,1)=4.
///
/// The first iteration folds Add into const 5, then the Sub sees two constants
/// and folds in the next iteration.
#[test]
fn test_constant_fold_chain() {
    let graph = load_onnx_simplified("constant_fold_chain.onnx");

    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Add { .. })),
        "Add should be folded"
    );
    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Sub { .. })),
        "Sub should be folded"
    );
    assert_eq!(
        count_operation_nodes(&graph),
        0,
        "only constants should remain"
    );
}

/// Dynamic input blocks folding: Neg(const) folds, but Add(dynamic, const) does not.
#[test]
fn test_constant_fold_blocked_by_dynamic() {
    let graph = load_onnx_simplified("constant_fold_blocked.onnx");

    // Neg(-5) -> 5 should be folded (all inputs constant)
    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Neg { .. })),
        "Neg on constant should be folded"
    );

    // Add(x, 5) must remain because x is dynamic
    assert!(
        has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Add { .. })),
        "Add with dynamic input should NOT be folded"
    );
}

/// Concat of constant shape slices folds into a single constant array.
///
/// Shape(x)[0:2]=[2,3], Shape(y)[0:1]=[5] => constant_shape folds the Slices
/// Concat([2,3], [5])=[2,3,5] => constant_fold folds the Concat
#[test]
fn test_constant_fold_concat() {
    let graph = load_onnx_simplified("constant_fold_concat.onnx");

    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Concat { .. })),
        "Concat of constants should be folded"
    );
    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Slice { .. })),
        "Slice on Shape should be folded by constant_shape"
    );
    assert_eq!(
        count_operation_nodes(&graph),
        0,
        "only constants should remain"
    );
}
