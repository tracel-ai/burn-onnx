//! # Custom (non-built-in) operators
//!
//! Nodes whose `(domain, op_type)` does not map to a built-in `NodeType` are
//! preserved as `NodeType::Custom` / `Node::Custom(CustomNode)` instead of
//! failing the parse. Type inference for them is supplied by user hooks
//! registered in `burn-onnx` (see `DESIGN-CUSTOM-OPS.md`); without a hook, a
//! best-effort same-as-input fallback keeps the graph buildable for inspection.

use std::sync::Arc;

use crate::ir::{ArgType, Argument, Node, PublicAttributesOwned, RawNode};
use crate::processor::{NodeProcessor, NodeSpec, OutputPreferences, ProcessError};

/// Inclusive opset range a hook supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpsetRange {
    /// Lowest supported opset.
    pub min: usize,
    /// Highest supported opset (`None` = unbounded).
    pub max: Option<usize>,
}

impl OpsetRange {
    /// Range from `min` with no upper bound.
    pub fn from_min(min: usize) -> Self {
        Self { min, max: None }
    }

    /// Whether `opset` falls inside the range.
    pub fn contains(&self, opset: usize) -> bool {
        opset >= self.min && self.max.is_none_or(|max| opset <= max)
    }
}

impl core::fmt::Display for OpsetRange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.max {
            Some(max) => write!(f, "{}..={max}", self.min),
            None => write!(f, "{}..", self.min),
        }
    }
}

/// Why a custom op is not covered by the registered hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MissingReason {
    /// No hook is registered for this `(op_type, domain)`.
    NoHook,
    /// A hook is registered but its opset range excludes the node's opset.
    OpsetMismatch {
        /// The opset range the registered hook supports.
        supported: OpsetRange,
    },
}

/// Coverage answer for a `(op_type, domain, opset)` triple.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HookCoverage {
    /// A hook is registered and its opset range covers the node.
    Covered,
    /// The node is not covered, with the reason for diagnostics.
    Missing(MissingReason),
}

/// Type-inference hooks for custom (non-built-in) operators.
///
/// Implemented by the hook registry in `burn-onnx` and passed to the parse
/// pipeline via `OnnxGraphBuilder::with_custom_op_inference`. Output
/// preferences from consumers are not consulted for custom ops: the hook is
/// the sole authority on its output types.
///
/// Contract: for any node, `coverage(..) == Covered` must imply
/// `infer(..)` returns `Ok(Some(_))`. The pipeline checks coverage for the
/// whole graph before inference runs, and treats `Ok(None)` after a passed
/// coverage check as an error rather than silently falling back.
pub trait CustomOpInference: Send + Sync {
    /// Coverage for this `(op_type, domain)` at the node's domain opset.
    fn coverage(&self, op_type: &str, domain: &str, opset: usize) -> HookCoverage;

    /// Infer output types. `Ok(None)` means no hook is registered for this
    /// node.
    ///
    /// Constant inputs are readable via `node.inputs[i].value()`. May be
    /// called more than once per node: type inference runs in an iterative
    /// fixed-point loop, so implementations must be deterministic and
    /// side-effect free.
    fn infer(&self, node: &CustomNode) -> Result<Option<Vec<ArgType>>, ProcessError>;
}

/// Public view of a custom (non-built-in) ONNX node.
///
/// Inputs are full [`Argument`] values with their value stores attached, so
/// constant input data is readable via `Argument::value()`.
///
/// During `infer_output_types` / [`CustomOpInference::infer`], the `outputs`
/// carry placeholder types: only their names and count are meaningful. Output
/// types are valid once inference has completed (i.e. in every codegen-side
/// hook method).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CustomNode {
    /// Node name (sanitized, unique within the graph)
    pub name: String,
    /// Raw ONNX op_type, e.g. "FftReal"
    pub op_type: String,
    /// Raw ONNX domain, e.g. "custom_domain" ("" = default domain)
    pub domain: String,
    /// The inputs of the node.
    pub inputs: Vec<Argument>,
    /// The outputs of the node.
    pub outputs: Vec<Argument>,
    /// ONNX attributes, exposed read-only.
    pub attrs: PublicAttributesOwned,
    /// Opset version of `domain` from the model's opset_import.
    pub opset: usize,
}

impl CustomNode {
    /// Construct a custom node view, e.g. for hook unit tests.
    ///
    /// Inside the pipeline, views are built from parsed nodes; hand-built
    /// nodes carry no attribute values or constant-input data unless the
    /// arguments were constructed with value stores.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        op_type: impl Into<String>,
        domain: impl Into<String>,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        attrs: PublicAttributesOwned,
        opset: usize,
    ) -> Self {
        Self {
            name: name.into(),
            op_type: op_type.into(),
            domain: domain.into(),
            inputs,
            outputs,
            attrs,
            opset,
        }
    }
}

impl core::fmt::Display for CustomNode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.domain.is_empty() {
            write!(f, "Custom({})", self.op_type)
        } else {
            write!(f, "Custom({}::{})", self.domain, self.op_type)
        }
    }
}

/// Build the public CustomNode view from a RawNode at any pipeline stage.
pub(crate) fn custom_node_view(node: &RawNode) -> CustomNode {
    let identity = node
        .custom_identity
        .as_ref()
        .expect("RawNode with NodeType::Custom must carry a CustomIdentity (parser invariant)");
    CustomNode {
        name: node.name.clone(),
        op_type: identity.op_type.clone(),
        domain: identity.domain.clone(),
        inputs: node.inputs.clone(),
        outputs: node.outputs.clone(),
        attrs: PublicAttributesOwned::from_internal(&node.attrs),
        opset: identity.domain_opset,
    }
}

/// Hook-free processor for `NodeType::Custom`, registered in the global registry.
///
/// Everything a processor must do for a custom node is hook-independent except
/// type inference: the spec is permissive, constants are never lifted, the node
/// is never a no-op, and `build_node` just snapshots the raw node into the
/// public `CustomNode` view. Type inference here is a best-effort fallback for
/// runs without registered hooks; the hook-aware path overrides it during the
/// type-inference phase.
pub(crate) struct CustomProcessor;

impl NodeProcessor for CustomProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        // Permissive: unknown schema, so any opset and any I/O count.
        NodeSpec::default()
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Best-effort fallback when no inference hook is in play. Types the
        // model declared (graph outputs / value_info, seeded during proto
        // conversion) are authoritative and never overwritten; only outputs
        // still carrying the placeholder default type get the guess.
        let placeholder = ArgType::default();
        if let Some(input_ty) = node.inputs.first().map(|input| input.ty.clone()) {
            let mut guessed = 0usize;
            for output in node
                .outputs
                .iter_mut()
                .filter(|output| output.ty == placeholder)
            {
                output.ty = input_ty.clone();
                guessed += 1;
            }
            if guessed > 0 {
                // debug, not warn: inference reruns in a fixed-point loop, so
                // this fires repeatedly; the once-per-node signal is the
                // "treating as custom op" log at proto conversion.
                log::debug!(
                    "No inference hook for custom op '{}'; assuming input type for {guessed} undeclared output(s)",
                    node.name
                );
            }
        } else if node.outputs.iter().any(|output| output.ty == placeholder) {
            log::debug!(
                "No inference hook for custom op '{}' and no inputs to mirror; keeping declared/default output types",
                node.name
            );
        }
        Ok(())
    }

    fn extract_config(&self, _node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Custom(custom_node_view(&builder))
    }
}

/// Hook-aware processor used by the type-inference phase in place of the
/// globally registered [`CustomProcessor`]; behaves identically to it when no
/// hooks are registered.
///
/// Only `infer_types` differs from the hook-free processor; every other
/// `NodeProcessor` responsibility is hook-independent.
pub(crate) struct HookedCustomProcessor {
    hooks: Option<Arc<dyn CustomOpInference>>,
}

impl HookedCustomProcessor {
    pub(crate) fn new(hooks: Option<Arc<dyn CustomOpInference>>) -> Self {
        Self { hooks }
    }

    /// The registered inference hook, if any.
    pub(crate) fn hooks(&self) -> Option<Arc<dyn CustomOpInference>> {
        self.hooks.clone()
    }
}

impl NodeProcessor for HookedCustomProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec::default()
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let Some(hooks) = &self.hooks else {
            // No hooks registered at all: hook-free fallback.
            return CustomProcessor.infer_types(node, opset, output_preferences);
        };

        let view = custom_node_view(node);
        let identity = format!(
            "custom op '{}' ({})",
            node.name,
            if view.domain.is_empty() {
                view.op_type.clone()
            } else {
                format!("{}::{}", view.domain, view.op_type)
            }
        );

        let inferred = hooks
            .infer(&view)
            .map_err(|e| ProcessError::Custom(format!("{identity}: {e}")))?;

        match inferred {
            Some(types) => {
                if types.len() != node.outputs.len() {
                    return Err(ProcessError::Custom(format!(
                        "{identity}: hook returned {} output type(s) but the node has {} output(s)",
                        types.len(),
                        node.outputs.len(),
                    )));
                }
                for (out, ty) in node.outputs.iter_mut().zip(types) {
                    out.ty = ty;
                }
                Ok(())
            }
            // The coverage pre-pass already verified this node is covered, so
            // a hook declining to infer here is a coverage()/infer() contract
            // violation, not a reason to silently fall back to a guess.
            None => Err(ProcessError::Custom(format!(
                "{identity}: coverage() reported the op covered but infer() declined to infer \
                 (contract violation in the CustomOpInference implementation)",
            ))),
        }
    }

    fn extract_config(&self, _node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        CustomProcessor.build_node(builder, opset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, CustomIdentity, DType, NodeType, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    fn make_custom_node() -> RawNode {
        let mut node = TestNodeBuilder::new(NodeType::Custom, "test_custom")
            .input_tensor_f32("input", 3, None)
            .output_tensor_f32("output", 0, None)
            .attr_int("n_fft", 1024)
            .build();
        node.custom_identity = Some(CustomIdentity {
            op_type: "FftReal".to_string(),
            domain: "custom_domain".to_string(),
            domain_opset: 2,
        });
        node
    }

    #[test]
    fn infer_types_falls_back_to_same_as_input() {
        let mut node = make_custom_node();
        let processor = CustomProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn infer_types_tolerates_no_inputs() {
        let mut node = make_custom_node();
        node.inputs.clear();
        let processor = CustomProcessor;
        // Must not panic on the empty-input case.
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
    }

    #[test]
    fn build_node_preserves_identity_and_attrs() {
        let node = make_custom_node();
        let built = CustomProcessor.build_node(node, 16);
        let Node::Custom(custom) = built else {
            panic!("expected Node::Custom");
        };
        assert_eq!(custom.op_type, "FftReal");
        assert_eq!(custom.domain, "custom_domain");
        assert_eq!(custom.opset, 2);
        assert_eq!(custom.attrs.get_i64("n_fft"), Some(1024));
        assert_eq!(custom.name, "test_custom");
        assert_eq!(custom.inputs.len(), 1);
        assert!(matches!(
            custom.inputs[0].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 3,
                ..
            })
        ));
    }

    #[test]
    fn node_type_accessor_and_display() {
        let built = CustomProcessor.build_node(make_custom_node(), 16);
        assert_eq!(built.node_type(), NodeType::Custom);
        assert_eq!(built.to_string(), "Custom(custom_domain::FftReal)");
    }

    #[test]
    fn from_str_never_resolves_to_custom() {
        use core::str::FromStr;
        assert!(NodeType::from_str("Custom").is_err());
        assert!(NodeType::from_str("custom").is_err());
    }

    /// Inference stub that returns a fixed set of output types for "FftReal".
    struct FixedInference {
        types: Vec<ArgType>,
    }

    impl CustomOpInference for FixedInference {
        fn coverage(&self, op_type: &str, _domain: &str, _opset: usize) -> HookCoverage {
            if op_type == "FftReal" {
                HookCoverage::Covered
            } else {
                HookCoverage::Missing(MissingReason::NoHook)
            }
        }

        fn infer(&self, node: &CustomNode) -> Result<Option<Vec<ArgType>>, ProcessError> {
            if node.op_type == "FftReal" {
                Ok(Some(self.types.clone()))
            } else {
                Ok(None)
            }
        }
    }

    #[test]
    fn hooked_processor_applies_hook_types() {
        let mut node = make_custom_node();
        let hook_ty = ArgType::Tensor(TensorType::new(DType::F64, 5, None));
        let processor = HookedCustomProcessor::new(Some(Arc::new(FixedInference {
            types: vec![hook_ty.clone()],
        })));
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        assert_eq!(node.outputs[0].ty, hook_ty);
    }

    #[test]
    fn hooked_processor_rejects_output_count_mismatch() {
        let mut node = make_custom_node();
        let ty = ArgType::Tensor(TensorType::new(DType::F32, 1, None));
        let processor = HookedCustomProcessor::new(Some(Arc::new(FixedInference {
            types: vec![ty.clone(), ty],
        })));
        let err = processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("returned 2 output type(s)"), "got: {msg}");
        assert!(msg.contains("custom_domain::FftReal"), "got: {msg}");
    }

    #[test]
    fn hooked_processor_errors_when_covered_hook_declines_to_infer() {
        // With hooks registered, reaching inference means the coverage
        // pre-pass passed; a hook then returning Ok(None) is a
        // coverage()/infer() contract violation, not a fallback case.
        let mut node = make_custom_node();
        node.custom_identity.as_mut().unwrap().op_type = "OtherOp".to_string();
        let processor =
            HookedCustomProcessor::new(Some(Arc::new(FixedInference { types: vec![] })));
        let err = processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("contract violation"), "got: {msg}");
        assert!(msg.contains("custom_domain::OtherOp"), "got: {msg}");
    }

    #[test]
    fn hooked_processor_wraps_hook_errors_with_op_identity() {
        struct FailingInference;

        impl CustomOpInference for FailingInference {
            fn coverage(&self, _: &str, _: &str, _: usize) -> HookCoverage {
                HookCoverage::Covered
            }

            fn infer(&self, _: &CustomNode) -> Result<Option<Vec<ArgType>>, ProcessError> {
                Err(ProcessError::MissingAttribute("n_fft".to_string()))
            }
        }

        let mut node = make_custom_node();
        let err = HookedCustomProcessor::new(Some(Arc::new(FailingInference)))
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("custom_domain::FftReal"), "got: {msg}");
        assert!(msg.contains("n_fft"), "got: {msg}");
    }

    #[test]
    fn hooked_processor_applies_types_to_multiple_outputs() {
        let mut node = make_custom_node();
        node.outputs
            .push(crate::ir::Argument::new("output2", ArgType::default()));
        let ty0 = ArgType::Tensor(TensorType::new(DType::F64, 2, None));
        let ty1 = ArgType::Tensor(TensorType::new(DType::I64, 1, None));
        HookedCustomProcessor::new(Some(Arc::new(FixedInference {
            types: vec![ty0.clone(), ty1.clone()],
        })))
        .infer_types(&mut node, 16, &OutputPreferences::new())
        .unwrap();
        assert_eq!(node.outputs[0].ty, ty0);
        assert_eq!(node.outputs[1].ty, ty1);
    }

    #[test]
    fn hooked_processor_with_no_hooks_matches_hook_free() {
        let mut node = make_custom_node();
        HookedCustomProcessor::new(None)
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn fallback_preserves_declared_output_types() {
        // A type the model declared (e.g. via value_info) must not be
        // clobbered by the same-as-input guess.
        let mut node = make_custom_node();
        let declared = ArgType::Tensor(TensorType::new(DType::I64, 4, None));
        node.outputs[0].ty = declared.clone();
        CustomProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        assert_eq!(node.outputs[0].ty, declared);
    }

    #[test]
    fn fallback_fills_all_undeclared_outputs() {
        let mut node = make_custom_node();
        node.outputs
            .push(crate::ir::Argument::new("output2", ArgType::default()));
        CustomProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
        assert_eq!(node.outputs[1].ty, node.inputs[0].ty);
    }
}
