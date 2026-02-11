use std::path::PathBuf;

pub use onnx_ir::ir::{Node, OnnxGraph};

pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/opset_compliance")
        .join(name)
}

pub fn load_model(name: &str) -> OnnxGraph {
    onnx_ir::OnnxGraphBuilder::new()
        .parse_file(&fixture_path(name))
        .unwrap_or_else(|e| panic!("Failed to parse '{name}': {e}"))
}

pub fn load_model_result(name: &str) -> Result<OnnxGraph, onnx_ir::Error> {
    onnx_ir::OnnxGraphBuilder::new().parse_file(&fixture_path(name))
}

/// Find a node whose name matches "prefix" followed by a digit.
/// This avoids `constant` matching `constantofshape1`.
pub fn find_node<'a>(graph: &'a OnnxGraph, name_prefix: &str) -> &'a Node {
    graph
        .nodes
        .iter()
        .find(|n| {
            let name = n.name();
            name.starts_with(name_prefix)
                && name[name_prefix.len()..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit())
        })
        .unwrap_or_else(|| {
            let names: Vec<_> = graph.nodes.iter().map(|n| n.name().to_string()).collect();
            panic!("No node with prefix '{name_prefix}'. Available: {names:?}")
        })
}
