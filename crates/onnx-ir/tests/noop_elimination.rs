/// Tests for no-op node elimination via the is_noop() processor trait method.
///
/// Verifies that nodes which produce output identical to their input
/// are automatically eliminated during post-processing.
mod test_utils;

use test_utils::*;

#[test]
fn test_cast_same_type_eliminated() {
    // Cast(F32 -> F32) is a no-op and should be eliminated (requires simplify=true)
    let graph = load_onnx_simplified("noop_cast_same_type.onnx");

    // Cast should be eliminated, only Relu remains
    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Cast { .. })),
        "Cast(F32->F32) should be eliminated as a no-op"
    );
    assert!(has_node_type(&graph, |n| matches!(
        n,
        onnx_ir::ir::Node::Relu { .. }
    )));
    assert_eq!(count_operation_nodes(&graph), 1);
}

#[test]
fn test_cast_different_type_preserved() {
    // Cast(F32 -> I64) is NOT a no-op and should be preserved
    let graph = load_onnx("noop_cast_different_type.onnx");

    assert!(
        has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Cast { .. })),
        "Cast(F32->I64) should NOT be eliminated"
    );
}

#[test]
fn test_identity_still_eliminated() {
    // Verify Identity elimination still works through the is_noop mechanism
    let graph = load_onnx("identity.onnx");

    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Identity { .. })),
        "Identity nodes should be eliminated via is_noop"
    );
}
