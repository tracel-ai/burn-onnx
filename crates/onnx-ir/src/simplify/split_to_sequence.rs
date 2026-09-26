use std::collections::{HashMap, HashSet};

use crate::TensorDataExt;
use crate::graph_state::GraphState;
use crate::ir::{
    ArgType, Argument, AttributeValue, Attributes, DType, NodeType, RawNode, ValueSource,
};
use crate::tensor_store::TensorDataRef;

/// Rewrite `SplitToSequence -> SequenceAt(constant position)*` into plain tensor ops.
///
/// Sequence values have no IR type, so both ops are unsupported on their own. When every
/// consumer of a `SplitToSequence` output is a `SequenceAt` with a constant position, each
/// read is a fixed chunk of the input and the sequence never has to exist:
///
/// - chunks of 1 with `keepdims=0`: `Gather(input, p, axis)`, dropping the axis
/// - any other chunk: `Slice` over its `[start, end)` range on the axis
///
/// "Chunks of 1" is no `split` input, or a scalar `split` of 1. The spec ignores
/// `keepdims` whenever `split` is given, but onnxruntime drops the axis for a scalar 1,
/// and that is how torch exports `unbind`, so the rewrite follows onnxruntime.
///
/// Each `SequenceAt` is rewritten in place, keeping its output, and the `SplitToSequence`
/// is removed. Runs before type inference, so it matches on constant values rather than
/// inferred types. Anything else (runtime positions or `split`, other sequence ops, the
/// sequence as a graph output, a negative position with a scalar `split`) is left
/// untouched and still fails as unsupported.
pub(crate) fn rewrite_split_to_sequence(state: &mut GraphState) {
    if !state
        .processed_nodes
        .iter()
        .any(|n| n.node_type == NodeType::SplitToSequence)
    {
        return;
    }
    let graph_outputs = state.node_graph_outputs();
    let mut nodes = std::mem::take(&mut state.processed_nodes);
    let removed = rewrite(&mut nodes, &graph_outputs, state);
    state.processed_nodes = nodes;
    state.remove_nodes(&removed);
}

/// Which chunk of the `SplitToSequence` input a `SequenceAt` reads.
enum Read {
    /// Index `p` along the axis, dropping it (`Gather`).
    Index(i64),
    /// The `[start, end)` range along the axis (`Slice`).
    Range(i64, i64),
}

/// Rewrites the readers in place and returns the (ascending) indices of the
/// `SplitToSequence` nodes left without consumers.
fn rewrite(
    nodes: &mut [RawNode],
    graph_outputs: &HashSet<String>,
    state: &mut GraphState,
) -> Vec<usize> {
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            consumers.entry(input.name.as_str()).or_default().push(i);
        }
    }

    // Resolve every reader before touching the graph, so a split with one unresolvable
    // reader is left whole.
    let mut resolved: Vec<(usize, Vec<(usize, Read)>)> = Vec::new();
    for (si, split) in nodes.iter().enumerate() {
        if split.node_type != NodeType::SplitToSequence {
            continue;
        }
        let seq = &split.outputs[0].name;
        let readers = consumers.get(seq.as_str()).map_or(&[][..], Vec::as_slice);
        let reads: Option<Vec<_>> = (!graph_outputs.contains(seq))
            .then(|| {
                readers
                    .iter()
                    .map(|&ri| resolve_read(split, seq, &nodes[ri]).map(|read| (ri, read)))
                    .collect()
            })
            .flatten();
        let Some(reads) = reads else {
            log::warn!(
                "Cannot rewrite '{}': its sequence must not be a graph output and must only \
                 be read by SequenceAt at constant positions (non-negative for a scalar split)",
                split.name
            );
            continue;
        };
        log::info!(
            "Rewriting '{}' and its {} SequenceAt reader(s) into tensor ops",
            split.name,
            reads.len()
        );
        resolved.push((si, reads));
    }

    let mut rewritten = Vec::new();
    for (si, reads) in &resolved {
        for (ri, read) in reads {
            nodes[*ri] = tensor_op(&nodes[*si], &nodes[*ri], read, state);
            rewritten.push(*ri);
        }
    }

    // Attached once all Slice bounds are stored: the first store copies the shared
    // tensor store, and a snapshot taken per bound would make every later store copy it.
    let value_store = state.build_value_store();
    for ri in rewritten {
        for input in nodes[ri]
            .inputs
            .iter_mut()
            .filter(|arg| arg.value_store.is_none())
        {
            input.set_value_store(value_store.clone());
        }
    }
    resolved.into_iter().map(|(si, _)| si).collect()
}

/// The chunk `reader` reads from `seq`, or `None` if it cannot be resolved statically.
fn resolve_read(split: &RawNode, seq: &str, reader: &RawNode) -> Option<Read> {
    if reader.node_type != NodeType::SequenceAt || reader.inputs[0].name != seq {
        return None;
    }
    let position = reader.get_input(1)?.value()?.to_i64_vec().ok()?;
    let &[p] = position.as_slice() else {
        return None;
    };

    let (sizes, scalar) = match split.get_input(1) {
        None => (vec![1], true),
        Some(arg) => {
            let value = arg.value()?;
            (value.to_i64_vec().ok()?, value.shape().is_empty())
        }
    };
    if scalar {
        let size = sizes[0];
        if size == 1 && int_attr(split, "keepdims").unwrap_or(1) == 0 {
            return Some(Read::Index(p));
        }
        if size == 1 {
            // `p + 1 == 0` would be an empty range, so the last chunk runs to the end.
            let end = if p == -1 { i64::MAX } else { p.checked_add(1)? };
            return Some(Read::Range(p, end));
        }
        // Chunks of `size`. Without the axis length, the chunk count (and so a negative
        // position) is unknown.
        if size <= 0 || p < 0 {
            return None;
        }
        let start = p.checked_mul(size)?;
        return Some(Read::Range(start, start.checked_add(size)?));
    }
    if sizes.iter().any(|&size| size < 0) {
        return None;
    }
    let n = sizes.len() as i64;
    let index = if p < 0 { p + n } else { p };
    if !(0..n).contains(&index) {
        return None;
    }
    let start = sizes[..index as usize]
        .iter()
        .try_fold(0i64, |sum, &size| sum.checked_add(size))?;
    Some(Read::Range(
        start,
        start.checked_add(sizes[index as usize])?,
    ))
}

/// The `Gather` or `Slice` replacing `reader`, keeping its name and output.
fn tensor_op(split: &RawNode, reader: &RawNode, read: &Read, state: &mut GraphState) -> RawNode {
    let axis = int_attr(split, "axis").unwrap_or(0);
    let data = split.inputs[0].clone();
    let (node_type, inputs, attrs) = match *read {
        Read::Index(p) => {
            let mut attrs = Attributes::new();
            attrs.insert("axis".to_string(), AttributeValue::Int64(axis));
            let index = static_i64(state, p, ArgType::ScalarNative(DType::I64));
            (NodeType::Gather, vec![data, index], attrs)
        }
        Read::Range(start, end) => {
            let bounds = [start, end, axis].map(|v| static_i64(state, v, ArgType::Shape(1)));
            let inputs = std::iter::once(data).chain(bounds).collect();
            (NodeType::Slice, inputs, Attributes::new())
        }
    };
    RawNode {
        custom_identity: None,
        node_type,
        name: reader.name.clone(),
        inputs,
        outputs: reader.outputs.clone(),
        attrs,
    }
}

/// A one-element i64 input stored in the graph's tensor store (value store attached
/// later): a scalar for a scalar `ty`, otherwise shape `[1]`.
fn static_i64(state: &mut GraphState, value: i64, ty: ArgType) -> Argument {
    let bytes = bytes::Bytes::copy_from_slice(&value.to_ne_bytes());
    let shape = if ty.is_scalar() { vec![] } else { vec![1] };
    let data_id = state.store_tensor_data(TensorDataRef::new(bytes, shape, DType::I64));
    Argument {
        name: String::new(),
        ty,
        value_source: ValueSource::Static(data_id),
        value_store: None,
    }
}

fn int_attr(node: &RawNode, name: &str) -> Option<i64> {
    match node.attrs.get(name) {
        Some(AttributeValue::Int64(v)) => Some(*v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplify::tests::{arg, node};

    fn split(inputs: Vec<Argument>) -> RawNode {
        let mut split = node("split", NodeType::SplitToSequence, &[], &["seq"]);
        split.inputs = inputs;
        split
    }

    fn reader(name: &str, position: Argument) -> RawNode {
        let mut reader = node(name, NodeType::SequenceAt, &["seq"], &[name]);
        reader.inputs.push(position);
        reader
    }

    #[test]
    fn unresolvable_patterns_are_left_intact() {
        let cases: Vec<(&str, Vec<RawNode>, &[&str])> = vec![
            (
                "runtime position",
                vec![split(vec![arg("x")]), reader("r", arg("p"))],
                &[],
            ),
            (
                "another sequence consumer",
                vec![
                    split(vec![arg("x")]),
                    reader("r", Argument::from_const_i64("", 0)),
                    node("len", NodeType::SequenceLength, &["seq"], &["n"]),
                ],
                &[],
            ),
            (
                "sequence is a graph output",
                vec![
                    split(vec![arg("x")]),
                    reader("r", Argument::from_const_i64("", 0)),
                ],
                &["seq"],
            ),
            (
                "runtime split",
                vec![
                    split(vec![arg("x"), arg("s")]),
                    reader("r", Argument::from_const_i64("", 0)),
                ],
                &[],
            ),
            (
                "negative position with a scalar split",
                vec![
                    split(vec![arg("x"), Argument::from_const_i64("", 2)]),
                    reader("r", Argument::from_const_i64("", -1)),
                ],
                &[],
            ),
        ];

        for (case, mut nodes, outputs) in cases {
            let outputs = outputs.iter().map(|s| s.to_string()).collect();
            let mut state = GraphState::new(&[], &[], &[], &[]);
            let removed = rewrite(&mut nodes, &outputs, &mut state);
            assert!(removed.is_empty(), "{case}");
            assert_eq!(nodes[1].node_type, NodeType::SequenceAt, "{case}");
        }
    }
}
