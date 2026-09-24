//! # Einsum
//!
//! Einstein summation over any number of operands.
//!
//! Supported forms follow the ONNX spec:
//! - explicit (`ij,jk->ik`) and implicit (`ij,jk`, output inferred alphabetically)
//! - any number of inputs (`ij->ji`, `ij,jk,kl->il`)
//! - repeated labels within one term for diagonals and traces (`ii->i`, `ii`)
//! - ellipsis broadcasting (`...ij,...jk->...ik`), including operands whose ellipsis
//!   widths differ, which broadcast right-aligned
//! - scalar operands with an empty or zero-width ellipsis term (`,ij->ij`, `...,ij->ij`)
//!
//! Labels are ASCII letters, upper or lower case.
//!
//! ONNX spec: <https://onnx.ai/onnx/operators/onnx__Einsum.html>

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// ONNX attributes for `Einsum`.
#[derive(Debug, Clone, Default)]
pub struct EinsumConfig {
    /// Equation string such as `bhwc,hkc->bhwk`, exactly as it appears in the model.
    pub equation: String,
}

/// IR node for `Einsum`.
#[derive(Debug, Clone, NodeBuilder)]
pub struct EinsumNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: EinsumConfig,
}

pub(crate) struct EinsumProcessor;

/// Einsum operands can arrive as regular tensors or rank-0 scalar values.
///
/// ONNX rank-0 values are represented in the IR as `ScalarNative` or `ScalarTensor`,
/// but type inference only cares about the effective rank, dtype, and any known
/// per-axis dimensions.
#[derive(Clone, Copy)]
struct EinsumOperand<'a> {
    dtype: DType,
    rank: usize,
    static_shape: Option<&'a [Option<usize>]>,
}

impl<'a> EinsumOperand<'a> {
    fn from_arg(arg: &'a ArgType) -> Result<Self, ProcessError> {
        match arg {
            ArgType::Tensor(tensor) => Ok(Self {
                dtype: tensor.dtype,
                rank: tensor.rank,
                static_shape: tensor.static_shape.as_deref(),
            }),
            ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => Ok(Self {
                dtype: *dtype,
                rank: 0,
                static_shape: Some(&[]),
            }),
            _ => Err(ProcessError::TypeMismatch {
                expected: "Tensor or scalar".to_string(),
                actual: format!("{arg:?}"),
            }),
        }
    }
}

impl NodeProcessor for EinsumProcessor {
    type Config = EinsumConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 12,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let equation = equation_attr(node)?;
        let parsed = ParsedEinsum::parse(&equation).map_err(ProcessError::Custom)?;

        let operands = node
            .inputs
            .iter()
            .map(|input| EinsumOperand::from_arg(&input.ty))
            .collect::<Result<Vec<_>, _>>()?;

        let dtype = operands[0].dtype;
        if dtype.is_bool() {
            return Err(ProcessError::TypeMismatch {
                expected: "numeric inputs".to_string(),
                actual: format!("{dtype:?}"),
            });
        }
        if let Some((index, operand)) = operands
            .iter()
            .enumerate()
            .find(|(_, operand)| operand.dtype != dtype)
        {
            return Err(ProcessError::TypeMismatch {
                expected: format!("all inputs to have dtype {dtype:?}"),
                actual: format!("input {index} has dtype {:?}", operand.dtype),
            });
        }

        let ranks: Vec<usize> = operands.iter().map(|operand| operand.rank).collect();
        let resolved = parsed
            .resolve(&ranks)
            .map_err(|reason| ProcessError::Custom(format!("Einsum '{equation}': {reason}")))?;
        let static_shape = infer_output_static_shape(&equation, &resolved, &operands)?;

        node.outputs[0].ty = if resolved.output.is_empty() {
            if node.inputs.iter().any(|input| input.ty.is_on_device()) {
                ArgType::ScalarTensor(dtype)
            } else {
                ArgType::ScalarNative(dtype)
            }
        } else {
            ArgType::Tensor(TensorType {
                dtype,
                rank: resolved.output.len(),
                static_shape,
            })
        };

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let equation = equation_attr(node)?;
        ParsedEinsum::parse(&equation).map_err(ProcessError::Custom)?;
        Ok(EinsumConfig { equation })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Einsum(EinsumNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

fn equation_attr(node: &RawNode) -> Result<String, ProcessError> {
    Ok(node
        .attrs
        .get("equation")
        .ok_or_else(|| ProcessError::MissingAttribute("equation".to_string()))?
        .clone()
        .into_string())
}

/// One subscript term: its named labels, and where `...` sits among them if present.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Term {
    labels: Vec<char>,
    ellipsis: Option<usize>,
}

impl Term {
    fn parse(term: &str, equation: &str) -> Result<Self, String> {
        let mut labels = Vec::new();
        let mut ellipsis = None;
        let mut rest = term;
        while let Some(c) = rest.chars().next() {
            if c == '.' {
                if !rest.starts_with("...") {
                    return Err(format!(
                        "Einsum equation '{equation}' has a '.' outside an ellipsis"
                    ));
                }
                if ellipsis.is_some() {
                    return Err(format!(
                        "Einsum equation '{equation}' has more than one ellipsis in term '{term}'"
                    ));
                }
                ellipsis = Some(labels.len());
                rest = &rest[3..];
            } else if c.is_ascii_alphabetic() {
                labels.push(c);
                rest = &rest[1..];
            } else {
                return Err(format!(
                    "Einsum equation '{equation}' contains invalid character '{c}'"
                ));
            }
        }
        Ok(Self { labels, ellipsis })
    }
}

/// An einsum equation parsed independently of operand ranks.
#[derive(Debug, Clone)]
struct ParsedEinsum {
    inputs: Vec<Term>,
    /// `None` in implicit form, where the output is derived from the inputs.
    output: Option<Term>,
}

/// One axis of a resolved term. Ellipsis axes are numbered within the broadcast
/// ellipsis block, so operands with narrower ellipses line up on the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Label {
    Ellipsis(usize),
    Named(char),
}

/// An equation bound to operand ranks: every term spelled out axis by axis.
#[derive(Debug)]
struct ResolvedEinsum {
    inputs: Vec<Vec<Label>>,
    output: Vec<Label>,
}

impl ParsedEinsum {
    fn parse(equation: &str) -> Result<Self, String> {
        let equation: String = equation.chars().filter(|c| !c.is_whitespace()).collect();

        let (inputs_str, output_str) = match equation.split_once("->") {
            Some((inputs, output)) => (inputs, Some(output)),
            None => (equation.as_str(), None),
        };

        let inputs = inputs_str
            .split(',')
            .map(|term| Term::parse(term, &equation))
            .collect::<Result<Vec<_>, _>>()?;
        let output = output_str
            .map(|term| Term::parse(term, &equation))
            .transpose()?;

        if let Some(output) = &output {
            let mut seen = std::collections::HashSet::new();
            for &c in &output.labels {
                if !seen.insert(c) {
                    return Err(format!(
                        "Einsum equation '{equation}' has repeated index '{c}' in the output"
                    ));
                }
                if !inputs.iter().any(|input| input.labels.contains(&c)) {
                    return Err(format!(
                        "Einsum equation '{equation}': output index '{c}' not found in any input"
                    ));
                }
            }
        }

        Ok(Self { inputs, output })
    }

    /// Spell out every term for the given operand ranks.
    fn resolve(&self, ranks: &[usize]) -> Result<ResolvedEinsum, String> {
        if ranks.len() != self.inputs.len() {
            return Err(format!(
                "{} input terms but {} inputs",
                self.inputs.len(),
                ranks.len()
            ));
        }

        // Each operand's ellipsis covers the axes its named labels leave over; the
        // broadcast ellipsis block is as wide as the widest of them.
        let mut widths = Vec::with_capacity(ranks.len());
        for (index, (term, &rank)) in self.inputs.iter().zip(ranks).enumerate() {
            let named = term.labels.len();
            let width = match term.ellipsis {
                Some(_) if rank >= named => rank - named,
                None if rank == named => 0,
                _ => {
                    return Err(format!(
                        "input {index} has rank {rank} but its term has {named} indices"
                    ));
                }
            };
            widths.push(width);
        }
        let width = widths.iter().copied().max().unwrap_or(0);

        let inputs: Vec<Vec<Label>> = self
            .inputs
            .iter()
            .zip(&widths)
            .map(|(term, &local)| expand_term(term, width - local, local))
            .collect();

        let output = match &self.output {
            Some(term) => expand_term(term, 0, width),
            // Implicit form: the ellipsis block, then every label that occurs exactly
            // once across the inputs, in alphabetical order.
            None => {
                let mut counts = std::collections::BTreeMap::new();
                for term in &self.inputs {
                    for &c in &term.labels {
                        *counts.entry(c).or_insert(0usize) += 1;
                    }
                }
                let has_ellipsis = self.inputs.iter().any(|term| term.ellipsis.is_some());
                (0..if has_ellipsis { width } else { 0 })
                    .map(Label::Ellipsis)
                    .chain(
                        counts
                            .into_iter()
                            .filter(|&(_, count)| count == 1)
                            .map(|(c, _)| Label::Named(c)),
                    )
                    .collect()
            }
        };

        Ok(ResolvedEinsum { inputs, output })
    }
}

/// Spell out a term whose ellipsis covers broadcast axes `offset..offset + width`.
fn expand_term(term: &Term, offset: usize, width: usize) -> Vec<Label> {
    let named = term.labels.iter().map(|&c| Label::Named(c));
    match term.ellipsis {
        None => named.collect(),
        Some(position) => {
            let mut labels: Vec<Label> = named.collect();
            labels.splice(
                position..position,
                (offset..offset + width).map(Label::Ellipsis),
            );
            labels
        }
    }
}

/// Known sizes of every output axis, checking that known sizes of a shared label
/// agree. A size of 1 broadcasts, so it never conflicts and is only reported when
/// every occurrence is known to be 1.
fn infer_output_static_shape(
    equation: &str,
    resolved: &ResolvedEinsum,
    operands: &[EinsumOperand<'_>],
) -> Result<Option<Vec<Option<usize>>>, ProcessError> {
    if operands
        .iter()
        .all(|operand| operand.static_shape.is_none())
    {
        return Ok(None);
    }

    /// What the occurrences of one label say about its size.
    #[derive(Default)]
    struct Evidence {
        /// The size of any occurrence known to be other than 1.
        size: Option<usize>,
        /// Whether some occurrence has an unknown size.
        unknown: bool,
    }

    let mut evidence = std::collections::BTreeMap::<Label, Evidence>::new();
    for (labels, operand) in resolved.inputs.iter().zip(operands) {
        // A label repeated within one term takes a diagonal, so its axes must be equal
        // exactly: a size of 1 only broadcasts against other operands.
        let mut in_term = std::collections::BTreeMap::<Label, usize>::new();
        for (axis, &label) in labels.iter().enumerate() {
            let Some(dim) = operand.static_shape.and_then(|shape| shape[axis]) else {
                continue;
            };
            if let Some(&first) = in_term.get(&label)
                && first != dim
            {
                return Err(ProcessError::Custom(format!(
                    "Einsum equation '{equation}' repeats {label:?} within one operand \
                     over axes of sizes {first} and {dim}"
                )));
            }
            in_term.insert(label, dim);
        }

        for (axis, &label) in labels.iter().enumerate() {
            let entry = evidence.entry(label).or_default();
            match operand.static_shape.and_then(|shape| shape[axis]) {
                None => entry.unknown = true,
                Some(1) => {}
                Some(dim) => match entry.size {
                    Some(known) if known != dim => {
                        return Err(ProcessError::Custom(format!(
                            "Einsum equation '{equation}' has mismatched static dimensions \
                             for {label:?}: {known} and {dim}"
                        )));
                    }
                    _ => entry.size = Some(dim),
                },
            }
        }
    }

    // A known size other than 1 wins, since every other occurrence either matches it
    // or broadcasts. Otherwise an unknown occurrence could still be anything.
    Ok(Some(
        resolved
            .output
            .iter()
            .map(|label| match evidence.get(label) {
                Some(Evidence {
                    size: Some(size), ..
                }) => Some(*size),
                Some(Evidence { unknown: false, .. }) => Some(1),
                _ => None,
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(equation: &str, ranks: &[usize]) -> RawNode {
        let shapes = vec![None; ranks.len()];
        create_test_node_with_partial_shapes(equation, ranks, shapes)
    }

    fn create_test_node_with_shapes(equation: &str, shapes: &[Vec<usize>]) -> RawNode {
        let ranks: Vec<usize> = shapes.iter().map(Vec::len).collect();
        let shapes = shapes
            .iter()
            .map(|shape| Some(shape.iter().copied().map(Some).collect()))
            .collect();
        create_test_node_with_partial_shapes(equation, &ranks, shapes)
    }

    fn create_test_node_with_partial_shapes(
        equation: &str,
        ranks: &[usize],
        shapes: Vec<Option<Vec<Option<usize>>>>,
    ) -> RawNode {
        let types = ranks
            .iter()
            .zip(shapes)
            .map(|(&rank, static_shape)| {
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank,
                    static_shape,
                })
            })
            .collect();
        create_test_node_with_types(equation, types)
    }

    fn create_test_node_with_types(equation: &str, types: Vec<ArgType>) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::Einsum, "test_einsum");
        for (index, ty) in types.into_iter().enumerate() {
            builder = builder.add_input(&format!("input{index}"), ty);
        }
        builder
            .output_tensor_f32("output", 0, None)
            .attr_string("equation", equation)
            .build()
    }

    fn infer(node: &mut RawNode) -> Result<(), ProcessError> {
        EinsumProcessor.infer_types(node, 16, &OutputPreferences::new())
    }

    fn output_tensor(node: &RawNode) -> &TensorType {
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            other => panic!("Expected tensor output, got {other:?}"),
        }
    }

    fn resolve(equation: &str, ranks: &[usize]) -> ResolvedEinsum {
        ParsedEinsum::parse(equation)
            .unwrap()
            .resolve(ranks)
            .unwrap()
    }

    fn named(labels: &str) -> Vec<Label> {
        labels.chars().map(Label::Named).collect()
    }

    #[test]
    fn test_parse_explicit_and_implicit() {
        let explicit = resolve("bhwc,hkc->bhwk", &[4, 3]);
        assert_eq!(explicit.inputs, vec![named("bhwc"), named("hkc")]);
        assert_eq!(explicit.output, named("bhwk"));

        // j appears twice (summed out); i and k appear once.
        assert_eq!(resolve("ij,jk", &[2, 2]).output, named("ik"));
        assert_eq!(resolve("ij,kl", &[2, 2]).output, named("ijkl"));
        assert_eq!(resolve("ij,ij", &[2, 2]).output, Vec::new());
    }

    #[test]
    fn test_parse_whitespace_tolerance() {
        let resolved = resolve("ij, jk -> ik", &[2, 2]);
        assert_eq!(resolved.inputs, vec![named("ij"), named("jk")]);
        assert_eq!(resolved.output, named("ik"));
    }

    #[test]
    fn test_parse_uppercase_labels() {
        // Uppercase sorts before lowercase in implicit output, as in numpy.
        let resolved = resolve("bA,Ac", &[2, 2]);
        assert_eq!(resolved.output, named("bc"));
        assert_eq!(resolve("aB,c", &[2, 1]).output, named("Bac"));
    }

    #[test]
    fn test_parse_any_number_of_inputs() {
        assert_eq!(resolve("ij->ji", &[2]).output, named("ji"));
        assert_eq!(resolve("ij,jk,kl->il", &[2, 2, 2]).output, named("il"));
    }

    #[test]
    fn test_parse_repeated_input_labels() {
        assert_eq!(resolve("ii->i", &[2]).output, named("i"));
        // Implicit trace: i occurs twice, so nothing survives.
        assert_eq!(resolve("ii", &[2]).output, Vec::new());
    }

    #[test]
    fn test_parse_rejects_invalid_equations() {
        assert!(ParsedEinsum::parse("i1,jk->ik").is_err());
        assert!(ParsedEinsum::parse("ij,jk->iz").is_err());
        assert!(ParsedEinsum::parse("ij,jk->ii").is_err());
        assert!(ParsedEinsum::parse("i..j->ij").is_err());
        assert!(ParsedEinsum::parse("...i...->i").is_err());
    }

    #[test]
    fn test_resolve_rejects_rank_mismatch() {
        let parsed = ParsedEinsum::parse("ij,jk->ik").unwrap();
        assert!(parsed.resolve(&[2, 3]).is_err());
        assert!(parsed.resolve(&[2]).is_err());
        let parsed = ParsedEinsum::parse("...ij->ij").unwrap();
        assert!(parsed.resolve(&[1]).is_err());
        assert!(parsed.resolve(&[2]).is_ok());
    }

    #[test]
    fn test_resolve_ellipsis() {
        let resolved = resolve("...ij,...jk->...ik", &[4, 4]);
        let batch = [Label::Ellipsis(0), Label::Ellipsis(1)];
        assert_eq!(resolved.output[..2], batch);
        assert_eq!(resolved.output[2..], named("ik")[..]);

        // Zero-width ellipsis.
        assert_eq!(resolve("...ij,...jk->...ik", &[2, 2]).output, named("ik"));

        // Implicit form puts the ellipsis first.
        let implicit = resolve("...ij,...jk", &[3, 3]);
        assert_eq!(implicit.output[0], Label::Ellipsis(0));
        assert_eq!(implicit.output[1..], named("ik")[..]);
    }

    #[test]
    fn test_resolve_ellipsis_broadcasts_right_aligned() {
        // The rank-3 operand's single ellipsis axis lines up with the last axis of the
        // rank-4 operand's two-axis ellipsis.
        let resolved = resolve("...ij,...jk->...ik", &[4, 3]);
        assert_eq!(resolved.inputs[1][0], Label::Ellipsis(1));
        assert_eq!(resolved.output.len(), 4);
    }

    #[test]
    fn test_resolve_scalar_with_zero_width_ellipsis() {
        let resolved = resolve("...,ij->ij", &[0, 2]);
        assert_eq!(resolved.inputs[0], Vec::new());
        assert_eq!(resolved.output, named("ij"));
    }

    #[test]
    fn test_infer_types_matmul() {
        let mut node = create_test_node("ij,jk->ik", &[2, 2]);
        infer(&mut node).unwrap();
        let tensor = output_tensor(&node);
        assert_eq!(tensor.dtype, DType::F32);
        assert_eq!(tensor.rank, 2);
    }

    #[test]
    fn test_infer_types_single_input() {
        let mut node = create_test_node_with_shapes("ij->i", &[vec![3, 4]]);
        infer(&mut node).unwrap();
        assert_eq!(output_tensor(&node).static_shape, Some(vec![Some(3)]));
    }

    #[test]
    fn test_infer_types_three_inputs() {
        let mut node =
            create_test_node_with_shapes("ij,jk,kl->il", &[vec![2, 3], vec![3, 4], vec![4, 5]]);
        infer(&mut node).unwrap();
        assert_eq!(
            output_tensor(&node).static_shape,
            Some(vec![Some(2), Some(5)])
        );
    }

    #[test]
    fn test_infer_types_batch_diagonal() {
        let mut node = create_test_node_with_shapes("...ii->...i", &[vec![3, 5, 5]]);
        infer(&mut node).unwrap();
        assert_eq!(
            output_tensor(&node).static_shape,
            Some(vec![Some(3), Some(5)])
        );
    }

    #[test]
    fn test_infer_types_trace_is_scalar() {
        let mut node = create_test_node("ii", &[2]);
        infer(&mut node).unwrap();
        assert!(matches!(
            node.outputs[0].ty,
            ArgType::ScalarTensor(DType::F32)
        ));
    }

    #[test]
    fn test_infer_types_rank_mismatch() {
        let mut node = create_test_node("bhwc,hkc->bhwk", &[3, 3]);
        assert!(infer(&mut node).is_err());
    }

    #[test]
    fn test_infer_types_dtype_mismatch() {
        let mut node = create_test_node_with_types(
            "ij,jk->ik",
            vec![
                ArgType::Tensor(TensorType::new_known(DType::F32, vec![2, 2])),
                ArgType::Tensor(TensorType::new_known(DType::F64, vec![2, 2])),
            ],
        );
        assert!(matches!(
            infer(&mut node),
            Err(ProcessError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_infer_types_rejects_bool() {
        let bool_tensor = || {
            ArgType::Tensor(TensorType::new_known(
                DType::Bool(crate::ir::BoolStore::Native),
                vec![2],
            ))
        };
        let mut node = create_test_node_with_types("i,i->i", vec![bool_tensor(), bool_tensor()]);
        assert!(matches!(
            infer(&mut node),
            Err(ProcessError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_infer_types_rejects_mismatched_static_dimensions() {
        let mut node = create_test_node_with_shapes("ij,jk->ik", &[vec![2, 3], vec![4, 5]]);
        assert!(matches!(infer(&mut node), Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_infer_types_partial_static_shapes() {
        let mut node = create_test_node_with_partial_shapes(
            "abij,abjk->abik",
            &[4, 4],
            vec![
                Some(vec![Some(2), None, Some(4), Some(5)]),
                Some(vec![Some(2), None, Some(5), Some(7)]),
            ],
        );
        infer(&mut node).unwrap();
        assert_eq!(
            output_tensor(&node).static_shape,
            Some(vec![Some(2), None, Some(4), Some(7)])
        );
    }

    #[test]
    fn test_infer_types_uses_any_known_dim_for_shared_axis() {
        let mut node = create_test_node_with_partial_shapes(
            "bij,bjk->bik",
            &[3, 3],
            vec![
                Some(vec![None, Some(3), Some(4)]),
                Some(vec![Some(2), Some(4), Some(5)]),
            ],
        );
        infer(&mut node).unwrap();
        assert_eq!(
            output_tensor(&node).static_shape,
            Some(vec![Some(2), Some(3), Some(5)])
        );
    }

    #[test]
    fn test_infer_types_ellipsis_broadcast_static_shape() {
        // The size-1 ellipsis axis of the first operand broadcasts against 3.
        let mut node =
            create_test_node_with_shapes("...ij,...jk->...ik", &[vec![1, 4, 5], vec![3, 5, 7]]);
        infer(&mut node).unwrap();
        assert_eq!(
            output_tensor(&node).static_shape,
            Some(vec![Some(3), Some(4), Some(7)])
        );
    }

    #[test]
    fn test_infer_types_rejects_diagonal_size_mismatch() {
        // Both axes of a diagonal must match; a size of 1 does not broadcast within one
        // operand.
        let mut node = create_test_node_with_shapes("ii->i", &[vec![1, 3]]);
        assert!(matches!(
            infer(&mut node),
            Err(ProcessError::Custom(msg)) if msg.contains("repeats")
        ));
    }

    #[test]
    fn test_infer_types_repeated_label_still_broadcasts_across_operands() {
        let mut node = create_test_node_with_shapes("ii,i->i", &[vec![3, 3], vec![1]]);
        infer(&mut node).unwrap();
        assert_eq!(output_tensor(&node).static_shape, Some(vec![Some(3)]));
    }

    #[test]
    fn test_infer_types_accepts_scalar_operands() {
        let mut node = create_test_node_with_types(
            ",ij->ij",
            vec![
                ArgType::ScalarNative(DType::F32),
                ArgType::Tensor(TensorType::new_known(DType::F32, vec![3, 4])),
            ],
        );
        infer(&mut node).unwrap();
        assert_eq!(
            output_tensor(&node).static_shape,
            Some(vec![Some(3), Some(4)])
        );

        let mut node = create_test_node_with_types(
            "ij,->ij",
            vec![
                ArgType::Tensor(TensorType::new_known(DType::F32, vec![3, 4])),
                ArgType::ScalarTensor(DType::F32),
            ],
        );
        infer(&mut node).unwrap();
        assert_eq!(output_tensor(&node).rank, 2);
    }

    #[test]
    fn test_infer_types_scalar_output_placement() {
        let mut node = create_test_node_with_types(
            ",->",
            vec![
                ArgType::ScalarNative(DType::F32),
                ArgType::ScalarNative(DType::F32),
            ],
        );
        infer(&mut node).unwrap();
        assert!(matches!(
            node.outputs[0].ty,
            ArgType::ScalarNative(DType::F32)
        ));

        let mut node = create_test_node_with_types(
            ",->",
            vec![
                ArgType::ScalarNative(DType::F32),
                ArgType::ScalarTensor(DType::F32),
            ],
        );
        infer(&mut node).unwrap();
        assert!(matches!(
            node.outputs[0].ty,
            ArgType::ScalarTensor(DType::F32)
        ));
    }

    #[test]
    fn test_extract_config_keeps_equation_verbatim() {
        let mut node = create_test_node("...ij,...jk->...ik", &[4, 4]);
        infer(&mut node).unwrap();
        let config = EinsumProcessor.extract_config(&node, 16).unwrap();
        assert_eq!(config.equation, "...ij,...jk->...ik");
    }
}
