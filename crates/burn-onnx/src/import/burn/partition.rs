use onnx_ir::{Argument, Node};
use std::collections::HashMap;
use std::collections::HashSet;

/// Minimum number of nodes before we consider partitioning at all.
pub(crate) const MIN_GRAPH_SIZE: usize = 200;

/// Minimum number of nodes in a single chunk (avoid tiny submodules).
const MIN_CHUNK_SIZE: usize = 64;

/// Maximum number of nodes in a single chunk before we force a split.
const MAX_CHUNK_SIZE: usize = 256;

/// Maximum cut width we'll accept for a partition point.
/// This counts all live values (tensors, scalars, shapes) crossing a boundary.
const MAX_CUT_WIDTH: usize = 64;

/// A partition of the node list into chunks, each becoming a submodule.
#[derive(Debug, Clone)]
pub(crate) struct Partition {
    /// Ranges of node indices for each chunk. E.g. [(0..50), (50..120), (120..200)]
    pub chunks: Vec<std::ops::Range<usize>>,
    /// For each chunk: the arguments that flow INTO the chunk from outside (inputs to its forward())
    pub chunk_inputs: Vec<Vec<Argument>>,
    /// For each chunk: the arguments that flow OUT of the chunk to later chunks or graph outputs
    pub chunk_outputs: Vec<Vec<Argument>>,
}

/// Compute the cut width at each position in the node list.
///
/// Position `p` is the boundary between node `p-1` and node `p`.
/// A value is "live" at position `p` if it was produced before `p` and consumed at or after `p`.
/// Counts all value types (Tensor, ScalarTensor, ScalarNative, Shape) so that partition points
/// minimize the total submodule interface width.
///
/// Returns a vector of length `nodes.len() + 1` where index `p` is the cut width at position `p`.
fn compute_cut_widths(nodes: &[Node], graph_output_names: &[String]) -> Vec<usize> {
    let n = nodes.len();

    // Map: value name -> (producer_position, last_consumer_position)
    // producer_position: 0 = graph input, 1..n = after node[i-1]
    // last_consumer_position: 0..n-1 = node[i], n = graph output
    let mut value_spans: HashMap<String, (usize, usize)> = HashMap::new();

    // Record graph output names so we can set last_consumer = n for them
    let graph_out_set: std::collections::HashSet<&str> =
        graph_output_names.iter().map(|s| s.as_str()).collect();

    // Walk nodes to find producer and last consumer positions.
    // We count ALL value types (Tensor, ScalarTensor, ScalarNative, Shape) so that
    // partition points are chosen where the total interface is narrow.
    for (i, node) in nodes.iter().enumerate() {
        // Record outputs as produced at position i+1
        for arg in node.outputs() {
            if !arg.name.is_empty() {
                value_spans
                    .entry(arg.name.clone())
                    .or_insert((i + 1, i + 1));
            }
        }

        // Record inputs as consumed at position i
        for arg in node.inputs() {
            if !arg.name.is_empty() && (arg.is_dynamic() || arg.is_constant()) {
                value_spans.entry(arg.name.clone()).and_modify(|(_, last)| {
                    if i > *last {
                        *last = i;
                    }
                });
                // If value is a graph input (not produced by any node), record it
                // with producer_position = 0
                value_spans.entry(arg.name.clone()).or_insert((0, i));
            }
        }
    }

    // Extend last_consumer to n for graph outputs
    for name in &graph_out_set {
        if let Some((_, last)) = value_spans.get_mut(*name) {
            *last = n;
        }
    }

    // Compute cut widths using a sweep-line (prefix sum) approach.
    // A value is live at position p if producer < p <= last_consumer.
    // We add +1 at producer+1 and -1 at last_consumer+1, then take the prefix sum.
    let mut delta = vec![0i64; n + 2]; // one extra for the -1 at the end
    for &(producer, last_consumer) in value_spans.values() {
        let start = producer + 1;
        let end = last_consumer + 1;
        if start <= n && start < end {
            delta[start] += 1;
            if end <= n {
                delta[end] -= 1;
            }
        }
    }

    let mut widths = vec![0usize; n + 1];
    let mut running: i64 = 0;
    for p in 0..=n {
        running += delta[p];
        debug_assert!(
            running >= 0,
            "cut width went negative at position {p}, running = {running}"
        );
        widths[p] = running.max(0) as usize;
    }

    widths
}

/// Find partition points using a greedy heuristic on cut widths.
///
/// Returns the positions at which to split (exclusive end of one chunk, start of next).
/// Returns empty if the graph is too small to partition.
fn find_partition_points(cut_widths: &[usize], node_count: usize) -> Vec<usize> {
    if node_count < MIN_GRAPH_SIZE {
        return vec![];
    }

    // Scan for candidate cut points with acceptable width
    // A candidate is a position p in [MIN_CHUNK_SIZE, node_count - MIN_CHUNK_SIZE]
    // where cut_widths[p] <= MAX_CUT_WIDTH
    let mut candidates: Vec<(usize, usize)> = Vec::new(); // (position, width)
    for (p, &w) in cut_widths
        .iter()
        .enumerate()
        .take(node_count.saturating_sub(MIN_CHUNK_SIZE) + 1)
        .skip(MIN_CHUNK_SIZE)
    {
        if w <= MAX_CUT_WIDTH {
            candidates.push((p, w));
        }
    }

    if candidates.is_empty() {
        log::warn!(
            "Graph has {node_count} nodes but no partition points with \
             cut width <= {MAX_CUT_WIDTH} were found; falling back to flat codegen"
        );
        return vec![];
    }

    // Greedy algorithm: pick the lowest-width cut in each region of size MAX_CHUNK_SIZE.
    // Start from position 0, look for the best cut in [MIN_CHUNK_SIZE, MAX_CHUNK_SIZE].
    // After picking a cut at position p, look for the next in [p + MIN_CHUNK_SIZE, p + MAX_CHUNK_SIZE].
    let mut points = Vec::new();
    let mut last_cut = 0;

    loop {
        let window_start = last_cut + MIN_CHUNK_SIZE;
        let window_end = (last_cut + MAX_CHUNK_SIZE).min(node_count);

        if window_start >= node_count.saturating_sub(MIN_CHUNK_SIZE) {
            // Remaining nodes would form too small a final chunk
            break;
        }

        // Find the lowest-width candidate in [window_start, window_end]
        let best = candidates
            .iter()
            .filter(|(p, _)| *p >= window_start && *p <= window_end)
            .min_by_key(|(_, w)| *w);

        if let Some(&(pos, _)) = best {
            points.push(pos);
            last_cut = pos;
        } else {
            // No acceptable cut in this window. Skip ahead and keep looking
            // rather than giving up entirely (flat codegen causes very slow
            // compilation for large models).
            log::debug!(
                "No acceptable partition point in nodes [{window_start}..{window_end}], \
                 skipping to next window"
            );
            last_cut = window_end;
        }
    }

    points
}

/// Determine the interface for each chunk: which tensors flow in and out.
fn compute_chunk_interfaces(
    nodes: &[Node],
    chunks: &[std::ops::Range<usize>],
    graph_input_args: &[Argument],
    graph_output_args: &[Argument],
) -> (Vec<Vec<Argument>>, Vec<Vec<Argument>>) {
    let num_chunks = chunks.len();
    let mut chunk_inputs = vec![Vec::new(); num_chunks];
    let mut chunk_outputs = vec![Vec::new(); num_chunks];

    // Map tensor name -> (chunk_index that produces it, Argument)
    let mut producers: HashMap<String, (usize, Argument)> = HashMap::new();

    // Register graph inputs as produced by "before chunk 0"
    for arg in graph_input_args {
        producers.insert(arg.name.clone(), (usize::MAX, arg.clone()));
    }

    // First pass: record which chunk produces each value
    for (chunk_idx, range) in chunks.iter().enumerate() {
        for node_idx in range.clone() {
            for arg in nodes[node_idx].outputs() {
                if !arg.name.is_empty() {
                    producers.insert(arg.name.clone(), (chunk_idx, arg.clone()));
                }
            }
        }
    }

    // Second pass: for each chunk, find tensors consumed from other chunks
    // Track which tensors are already added as inputs/outputs to avoid duplicates
    let mut chunk_input_sets: Vec<std::collections::HashSet<String>> =
        vec![std::collections::HashSet::new(); num_chunks];
    let mut chunk_output_sets: Vec<std::collections::HashSet<String>> =
        vec![std::collections::HashSet::new(); num_chunks];

    for (chunk_idx, range) in chunks.iter().enumerate() {
        for node_idx in range.clone() {
            for arg in nodes[node_idx].inputs() {
                if arg.name.is_empty() {
                    continue;
                }
                if !arg.is_dynamic() && !arg.is_constant() {
                    continue;
                }

                if let Some(&(producer_chunk, ref producer_arg)) = producers.get(&arg.name)
                    && producer_chunk != chunk_idx
                {
                    // This tensor comes from outside this chunk
                    if chunk_input_sets[chunk_idx].insert(arg.name.clone()) {
                        chunk_inputs[chunk_idx].push(producer_arg.clone());
                    }
                    // Mark it as an output of the producing chunk (if it's a real chunk)
                    if producer_chunk != usize::MAX
                        && chunk_output_sets[producer_chunk].insert(arg.name.clone())
                    {
                        chunk_outputs[producer_chunk].push(producer_arg.clone());
                    }
                }
            }
        }
    }

    // Mark graph outputs: if a graph output is produced by a chunk, it must be in that chunk's outputs
    for arg in graph_output_args {
        if let Some(&(producer_chunk, ref producer_arg)) = producers.get(&arg.name)
            && producer_chunk != usize::MAX
            && chunk_output_sets[producer_chunk].insert(arg.name.clone())
        {
            chunk_outputs[producer_chunk].push(producer_arg.clone());
        }
    }

    (chunk_inputs, chunk_outputs)
}

/// Reorder constant nodes so each appears just before its first consumer.
///
/// Constants have no data dependencies on other nodes, so repositioning them is always safe.
/// This prevents clusters of constants from creating partition boundaries with wide
/// interfaces (constants are struct fields and shouldn't be passed through `forward()`).
pub(crate) fn reorder_constants_to_consumers(nodes: &mut Vec<Node>) {
    let n = nodes.len();
    if n == 0 {
        return;
    }

    // Identify constant nodes and map their output names to node index.
    let mut is_constant = vec![false; n];
    let mut const_output_to_idx: HashMap<String, usize> = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
        if matches!(node, Node::Constant(_)) {
            is_constant[i] = true;
            for arg in node.outputs() {
                if !arg.name.is_empty() {
                    const_output_to_idx.insert(arg.name.clone(), i);
                }
            }
        }
    }

    if const_output_to_idx.is_empty() {
        return;
    }

    // Find the first consumer for each constant.
    let mut const_first_consumer: HashMap<usize, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for arg in node.inputs() {
            if let Some(&const_idx) = const_output_to_idx.get(&arg.name) {
                const_first_consumer.entry(const_idx).or_insert(i);
            }
        }
    }

    // Group constants by their first consumer node index.
    let mut consumer_to_constants: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut orphan_constants: Vec<usize> = Vec::new();

    for (i, &is_const) in is_constant.iter().enumerate() {
        if !is_const {
            continue;
        }
        match const_first_consumer.get(&i) {
            Some(&consumer) if consumer != i + 1 => {
                // Only reorder if the constant isn't already right before its consumer
                consumer_to_constants.entry(consumer).or_default().push(i);
            }
            _ => {
                // Already in place or orphan
                orphan_constants.push(i);
            }
        }
    }

    if consumer_to_constants.is_empty() {
        return; // Nothing to move
    }

    // Track which constants are being relocated.
    let relocated: HashSet<usize> = consumer_to_constants.values().flatten().copied().collect();

    // Build new node order: orphan constants and non-relocated nodes keep their
    // relative order; relocated constants are inserted before their first consumer.
    let mut new_order: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if relocated.contains(&i) {
            continue; // Will be inserted before its consumer
        }
        // Before inserting this non-constant node, prepend any constants that target it.
        if let Some(consts) = consumer_to_constants.get(&i) {
            new_order.extend(consts);
        }
        new_order.push(i);
    }

    debug_assert_eq!(new_order.len(), n);

    // Apply the reordering.
    let mut slots: Vec<Option<Node>> = nodes.drain(..).map(Some).collect();
    *nodes = new_order
        .into_iter()
        .map(|i| slots[i].take().expect("node used twice"))
        .collect();
}

/// Try to partition a graph's nodes into submodule chunks.
///
/// Returns `None` if the graph is too small to benefit from partitioning.
/// Returns `Some(Partition)` with chunk ranges and interfaces if partitioning is beneficial.
pub(crate) fn try_partition(
    nodes: &[Node],
    graph_input_args: &[Argument],
    graph_output_args: &[Argument],
) -> Option<Partition> {
    let n = nodes.len();
    if n < MIN_GRAPH_SIZE {
        return None;
    }

    let graph_output_names: Vec<String> =
        graph_output_args.iter().map(|a| a.name.clone()).collect();
    let cut_widths = compute_cut_widths(nodes, &graph_output_names);
    let points = find_partition_points(&cut_widths, n);

    if points.is_empty() {
        return None;
    }

    // Build chunk ranges from partition points
    let mut chunks = Vec::new();
    let mut start = 0;
    for &p in &points {
        chunks.push(start..p);
        start = p;
    }
    chunks.push(start..n);

    let (chunk_inputs, chunk_outputs) =
        compute_chunk_interfaces(nodes, &chunks, graph_input_args, graph_output_args);

    Some(Partition {
        chunks,
        chunk_inputs,
        chunk_outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_widths_empty_graph() {
        let widths = compute_cut_widths(&[], &[]);
        assert_eq!(widths, vec![0]);
    }

    #[test]
    fn partition_returns_none_for_small_graph() {
        assert!(try_partition(&[], &[], &[]).is_none());
    }

    #[test]
    fn find_partition_points_returns_empty_for_small() {
        let widths = vec![0; 50];
        assert!(find_partition_points(&widths, 49).is_empty());
    }

    #[test]
    fn find_partition_points_finds_narrow_cut() {
        // 300 nodes, low cut width everywhere (so many cuts are valid),
        // but the narrowest at position 150
        let mut widths = vec![5usize; 301];
        widths[150] = 1;
        let points = find_partition_points(&widths, 300);
        // Should include the narrow cut at 150
        assert!(points.contains(&150));
    }

    #[test]
    fn find_partition_points_multiple_cuts() {
        // 600 nodes, narrow cuts at 200, 400; everything else is acceptable (width 5)
        let mut widths = vec![5usize; 601];
        widths[200] = 1;
        widths[400] = 1;
        let points = find_partition_points(&widths, 600);
        assert!(points.contains(&200));
        assert!(points.contains(&400));
    }
}
