use std::collections::HashMap;

use crate::ir::{NodeType, RawNode, TensorDataExt};

/// Eliminate binary ops where one operand is an identity element.
///
/// Patterns:
/// - `x + 0 = x`, `0 + x = x`
/// - `x - 0 = x`
/// - `x * 1 = x`, `1 * x = x`
/// - `x / 1 = x`
/// - `x ** 1 = x`
///
/// When detected, rewires consumers to use the non-identity input directly.
/// The eliminated node becomes dead and is cleaned up by dead node elimination.
pub(crate) fn eliminate_identity_elements(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    let mut rename: HashMap<String, String> = HashMap::new();

    for node in &nodes {
        if node.inputs.len() != 2 || node.outputs.len() != 1 {
            continue;
        }

        let passthrough = match node.node_type {
            // x + 0 or 0 + x; x - 0
            NodeType::Add | NodeType::Sub => {
                if is_constant_value(&node.inputs[1], 0.0) {
                    Some(0) // keep input[0]
                } else if node.node_type == NodeType::Add && is_constant_value(&node.inputs[0], 0.0)
                {
                    Some(1) // keep input[1]
                } else {
                    None
                }
            }
            // x * 1 or 1 * x
            NodeType::Mul => {
                if is_constant_value(&node.inputs[1], 1.0) {
                    Some(0)
                } else if is_constant_value(&node.inputs[0], 1.0) {
                    Some(1)
                } else {
                    None
                }
            }
            // x / 1
            NodeType::Div => {
                if is_constant_value(&node.inputs[1], 1.0) {
                    Some(0)
                } else {
                    None
                }
            }
            // x ** 1
            NodeType::Pow => {
                if is_constant_value(&node.inputs[1], 1.0) {
                    Some(0)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(keep_idx) = passthrough {
            let keep_name = &node.inputs[keep_idx].name;
            log::debug!(
                "Identity element elimination: {:?} '{}' (output '{}' -> input '{}')",
                node.node_type,
                node.name,
                node.outputs[0].name,
                keep_name,
            );
            rename.insert(node.outputs[0].name.clone(), keep_name.clone());
        }
    }

    if rename.is_empty() {
        return nodes;
    }

    log::info!(
        "Identity element elimination: found {} redundant op(s)",
        rename.len()
    );

    for node in &mut nodes {
        for input in &mut node.inputs {
            if let Some(new_name) = rename.get(&input.name) {
                input.name = new_name.clone();
            }
        }
    }

    nodes
}

/// Check if an argument is a constant where all elements equal `target`.
///
/// Works for scalar constants and broadcast-compatible constant tensors.
fn is_constant_value(arg: &crate::ir::Argument, target: f64) -> bool {
    let data = match arg.value() {
        Some(d) => d,
        None => return false,
    };

    let values = match data.to_f32_vec() {
        Ok(v) => v,
        Err(_) => return false,
    };

    if values.is_empty() {
        return false;
    }

    let target_f32 = target as f32;
    values.iter().all(|&v| v == target_f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, TensorType, ValueSource};
    use crate::simplify::tests::node;
    use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};

    fn const_f32_arg(name: &str, value: f32) -> Argument {
        let bytes = bytes::Bytes::copy_from_slice(&value.to_ne_bytes());
        let data_ref = TensorDataRef::new(bytes, vec![1], DType::F32);
        let mut store = TensorStore::new();
        let id = store.store(data_ref);
        let mut constant_map = std::collections::HashMap::new();
        constant_map.insert(name.to_string(), id);
        let value_store = ValueStore::new(std::rc::Rc::new(store), std::rc::Rc::new(constant_map));
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 0,
                static_shape: Some(vec![]),
            }),
            value_source: ValueSource::Constant,
            value_store: Some(value_store),
        }
    }

    fn make_binary_node(
        name: &str,
        op: NodeType,
        lhs: Argument,
        rhs: Argument,
        out: &str,
    ) -> RawNode {
        RawNode {
            node_type: op,
            name: name.to_string(),
            inputs: vec![lhs, rhs],
            outputs: vec![crate::simplify::tests::arg(out)],
            attrs: Default::default(),
        }
    }

    #[test]
    fn test_add_zero_eliminated() {
        let x = crate::simplify::tests::arg("x");
        let zero = const_f32_arg("zero", 0.0);
        let nodes = vec![
            make_binary_node("add", NodeType::Add, x, zero, "add_out"),
            node("relu", NodeType::Relu, &["add_out"], &["output"]),
        ];

        let result = eliminate_identity_elements(nodes);
        let relu = result.iter().find(|n| n.name == "relu").unwrap();
        assert_eq!(relu.inputs[0].name, "x");
    }

    #[test]
    fn test_zero_plus_x_eliminated() {
        let zero = const_f32_arg("zero", 0.0);
        let x = crate::simplify::tests::arg("x");
        let nodes = vec![
            make_binary_node("add", NodeType::Add, zero, x, "add_out"),
            node("relu", NodeType::Relu, &["add_out"], &["output"]),
        ];

        let result = eliminate_identity_elements(nodes);
        let relu = result.iter().find(|n| n.name == "relu").unwrap();
        assert_eq!(relu.inputs[0].name, "x");
    }

    #[test]
    fn test_mul_one_eliminated() {
        let x = crate::simplify::tests::arg("x");
        let one = const_f32_arg("one", 1.0);
        let nodes = vec![
            make_binary_node("mul", NodeType::Mul, x, one, "mul_out"),
            node("relu", NodeType::Relu, &["mul_out"], &["output"]),
        ];

        let result = eliminate_identity_elements(nodes);
        let relu = result.iter().find(|n| n.name == "relu").unwrap();
        assert_eq!(relu.inputs[0].name, "x");
    }

    #[test]
    fn test_div_one_eliminated() {
        let x = crate::simplify::tests::arg("x");
        let one = const_f32_arg("one", 1.0);
        let nodes = vec![
            make_binary_node("div", NodeType::Div, x, one, "div_out"),
            node("relu", NodeType::Relu, &["div_out"], &["output"]),
        ];

        let result = eliminate_identity_elements(nodes);
        let relu = result.iter().find(|n| n.name == "relu").unwrap();
        assert_eq!(relu.inputs[0].name, "x");
    }

    #[test]
    fn test_pow_one_eliminated() {
        let x = crate::simplify::tests::arg("x");
        let one = const_f32_arg("one", 1.0);
        let nodes = vec![
            make_binary_node("pow", NodeType::Pow, x, one, "pow_out"),
            node("relu", NodeType::Relu, &["pow_out"], &["output"]),
        ];

        let result = eliminate_identity_elements(nodes);
        let relu = result.iter().find(|n| n.name == "relu").unwrap();
        assert_eq!(relu.inputs[0].name, "x");
    }

    #[test]
    fn test_sub_zero_eliminated() {
        let x = crate::simplify::tests::arg("x");
        let zero = const_f32_arg("zero", 0.0);
        let nodes = vec![
            make_binary_node("sub", NodeType::Sub, x, zero, "sub_out"),
            node("relu", NodeType::Relu, &["sub_out"], &["output"]),
        ];

        let result = eliminate_identity_elements(nodes);
        let relu = result.iter().find(|n| n.name == "relu").unwrap();
        assert_eq!(relu.inputs[0].name, "x");
    }

    #[test]
    fn test_add_nonzero_not_eliminated() {
        let x = crate::simplify::tests::arg("x");
        let val = const_f32_arg("val", 0.5);
        let nodes = vec![
            make_binary_node("add", NodeType::Add, x, val, "add_out"),
            node("relu", NodeType::Relu, &["add_out"], &["output"]),
        ];

        let result = eliminate_identity_elements(nodes);
        let relu = result.iter().find(|n| n.name == "relu").unwrap();
        assert_eq!(relu.inputs[0].name, "add_out");
    }

    #[test]
    fn test_dynamic_input_not_eliminated() {
        // Both inputs are dynamic (no constant value), should not be eliminated
        let nodes = vec![
            node("add", NodeType::Add, &["x", "y"], &["add_out"]),
            node("relu", NodeType::Relu, &["add_out"], &["output"]),
        ];

        let result = eliminate_identity_elements(nodes);
        let relu = result.iter().find(|n| n.name == "relu").unwrap();
        assert_eq!(relu.inputs[0].name, "add_out");
    }
}
