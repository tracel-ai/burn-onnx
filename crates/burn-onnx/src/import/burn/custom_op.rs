//! Custom operator codegen hooks.
//!
//! Users implement [`CustomOp`] for each non-built-in ONNX operator in their
//! model and register the implementations via `ModelGen::register_custom_op`.
//! The [`HookRegistry`] stores them and doubles as the `CustomOpInference`
//! implementation handed to the `onnx-ir` parse pipeline.

use std::collections::HashMap;

use onnx_ir::{
    ArgType, CustomNode, CustomOpInference, HookCoverage, MissingReason, Node, NodeType,
    OpsetRange, ProcessError,
};
use proc_macro2::TokenStream;

use crate::burn::node_traits::Field;
use crate::ext::{CodegenContext, Imports};
use burn_store::TensorSnapshot;

/// Codegen hook for one custom (non-built-in) ONNX operator.
///
/// A hook is matched by ONNX operator identity `(op_type, domain)` and is
/// responsible for both type inference (during parsing) and code generation.
/// Constant inputs are readable via `node.inputs[i].value()` in every method.
pub trait CustomOp: Send + Sync + 'static {
    /// ONNX op_type this hook handles (e.g. "FftReal").
    fn op_type(&self) -> &str;

    /// ONNX domain. Empty string = default ONNX domain.
    ///
    /// `""` and `"ai.onnx"` are the same domain per the ONNX spec and are
    /// canonicalized on registration, so either spelling matches a node either
    /// way round. Registering both for one op_type is a duplicate.
    fn domain(&self) -> &str {
        ""
    }

    /// Opset gate, checked against the node's domain opset by the coverage
    /// pre-pass. Out-of-range is reported as a missing hook.
    fn opset_range(&self) -> OpsetRange {
        OpsetRange::from_min(1)
    }

    /// Infer output ArgTypes. Called during onnx-ir type inference.
    ///
    /// Must return exactly `node.outputs.len()` types; the pipeline rejects a
    /// mismatch. Consumers' output preferences are not consulted for custom
    /// ops: this hook is the sole authority on its output types. May be
    /// called more than once per node (inference runs in a fixed-point
    /// loop), so it must be deterministic and side-effect free.
    fn infer_output_types(&self, node: &CustomNode) -> Result<Vec<ArgType>, ProcessError>;

    /// Generate the forward-pass code for this node.
    ///
    /// Code generation has no recoverable error channel: an `Err` here fails
    /// the build, with a message naming this op and the method that failed.
    /// It is still the right way to reject a configuration you cannot
    /// handle, because the failure is attributed to your hook instead of
    /// surfacing as a raw panic from inside it. Better still, reject in
    /// [`infer_output_types`](Self::infer_output_types), which fails during
    /// parsing alongside the other custom-op diagnostics.
    fn forward(
        &self,
        node: &CustomNode,
        ctx: &mut CodegenContext<'_, '_>,
    ) -> Result<TokenStream, ProcessError>;

    /// Optional: extra imports emitted as `use` statements in the model file.
    fn register_imports(&self, _imports: &mut Imports<'_>) {}

    /// Optional: declare a module field (e.g. learnable params or state).
    fn field(&self, _node: &CustomNode) -> Result<Option<Field>, ProcessError> {
        Ok(None)
    }

    /// Optional: weights/snapshot collection (parallels the built-in nodes).
    fn collect_snapshots(
        &self,
        _node: &CustomNode,
        _field_name: &str,
    ) -> Result<Vec<TensorSnapshot>, ProcessError> {
        Ok(vec![])
    }
}

/// Codegen override for one built-in ONNX operator.
///
/// The built-in processor still performs type inference (overrides are
/// codegen-only); the override replaces only the emitted code. It receives
/// the typed [`Node`] so it can downcast to the concrete node variant.
pub trait OpOverride: Send + Sync + 'static {
    /// The built-in node type to override, e.g. `NodeType::MatMul`.
    fn target(&self) -> NodeType;

    /// Generate the forward-pass code for this node in place of the built-in.
    ///
    /// Code generation has no recoverable error channel: an `Err` here fails
    /// the build, with a message naming the node and the method that failed.
    /// It is still the right way to reject a node you cannot handle (e.g. a
    /// variant your kernel does not support), because the failure is
    /// attributed to your override instead of surfacing as a raw panic from
    /// inside it.
    fn forward(
        &self,
        node: &Node,
        ctx: &mut CodegenContext<'_, '_>,
    ) -> Result<TokenStream, ProcessError>;

    /// Optional: extra imports emitted as `use` statements in the model file.
    fn register_imports(&self, _imports: &mut Imports<'_>) {}

    /// Optional: declare a module field in place of the built-in's.
    ///
    /// The default (`Ok(None)`) suppresses the built-in's field: overriding a
    /// weighted op means reimplementing both `field` and `collect_snapshots`.
    fn field(&self, _node: &Node) -> Result<Option<Field>, ProcessError> {
        Ok(None)
    }

    /// Optional: weights/snapshot collection in place of the built-in's.
    fn collect_snapshots(
        &self,
        _node: &Node,
        _field_name: &str,
    ) -> Result<Vec<TensorSnapshot>, ProcessError> {
        Ok(vec![])
    }
}

/// Registry of user codegen hooks, keyed by ONNX operator identity.
///
/// Owned by `ModelGen` (behind `Arc`), shared with the onnx-ir parse pipeline
/// as its `CustomOpInference` implementation and with `BurnGraph` for codegen
/// dispatch.
#[derive(Default)]
pub(crate) struct HookRegistry {
    customs: HashMap<(String, String), Box<dyn CustomOp>>,
    overrides: HashMap<NodeType, Box<dyn OpOverride>>,
}

impl std::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRegistry")
            .field("customs", &self.customs.keys().collect::<Vec<_>>())
            .field("overrides", &self.overrides.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Render an ONNX operator identity for diagnostics, matching how
/// `CustomNode`'s `Display` renders it: the default domain is the empty
/// string, so `domain::op_type` would print a bare leading `::`.
fn format_identity(op_type: &str, domain: &str) -> String {
    if domain.is_empty() {
        op_type.to_string()
    } else {
        format!("{domain}::{op_type}")
    }
}

impl HookRegistry {
    /// Register a custom op hook. Panics on duplicate `(op_type, domain)`:
    /// registration happens in build scripts, where an immediate, attributable
    /// panic beats a silently shadowed hook.
    pub(crate) fn add_custom_op(&mut self, op: Box<dyn CustomOp>) {
        // Canonicalize the declared domain to match how onnx-ir stores a node's
        // identity, so declaring "ai.onnx" and "" is a genuine duplicate rather
        // than two hooks of which only one can ever match.
        let key = (
            op.op_type().to_string(),
            onnx_ir::normalize_domain(op.domain()).to_string(),
        );
        if self.customs.contains_key(&key) {
            panic!(
                "Duplicate custom op registration for '{}'",
                format_identity(&key.0, &key.1)
            );
        }
        self.customs.insert(key, op);
    }

    /// Register a built-in op override. Panics on duplicate target and on
    /// `NodeType::Custom` (custom ops are handled by `add_custom_op`).
    pub(crate) fn add_override(&mut self, over: Box<dyn OpOverride>) {
        let target = over.target();
        if target == NodeType::Custom {
            panic!("OpOverride cannot target NodeType::Custom; register a CustomOp hook instead");
        }
        if self.overrides.contains_key(&target) {
            panic!("Duplicate op override registration for {target:?}");
        }
        self.overrides.insert(target, over);
    }

    /// Look up the hook for an ONNX operator identity.
    ///
    /// Linear scan instead of a keyed lookup: a `(String, String)` map key
    /// cannot be borrowed as `(&str, &str)`, so a HashMap would allocate two
    /// Strings per lookup, and registries hold a handful of hooks at most.
    pub(crate) fn custom_for(&self, op_type: &str, domain: &str) -> Option<&dyn CustomOp> {
        // Keys are canonicalized by `add_custom_op`, so canonicalize the probe
        // too: both spellings of the default domain must reach the same hook.
        let domain = onnx_ir::normalize_domain(domain);
        self.customs
            .iter()
            .find(|((t, d), _)| t == op_type && d == domain)
            .map(|(_, b)| b.as_ref())
    }

    /// Look up the override for a built-in node type.
    pub(crate) fn override_for(&self, node_type: &NodeType) -> Option<&dyn OpOverride> {
        self.overrides.get(node_type).map(|b| b.as_ref())
    }
}

impl CustomOpInference for HookRegistry {
    fn coverage(&self, op_type: &str, domain: &str, opset: usize) -> HookCoverage {
        match self.custom_for(op_type, domain) {
            None => HookCoverage::Missing(MissingReason::NoHook),
            Some(op) => {
                let supported = op.opset_range();
                if supported.contains(opset) {
                    HookCoverage::Covered
                } else {
                    HookCoverage::Missing(MissingReason::OpsetMismatch { supported })
                }
            }
        }
    }

    fn infer(&self, node: &CustomNode) -> Result<Option<Vec<ArgType>>, ProcessError> {
        match self.custom_for(&node.op_type, &node.domain) {
            Some(op) => op.infer_output_types(node).map(Some),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::ir::{DType, TensorType};

    struct TestOp;

    impl CustomOp for TestOp {
        fn op_type(&self) -> &str {
            "FftReal"
        }

        fn domain(&self) -> &str {
            "custom_domain"
        }

        fn opset_range(&self) -> OpsetRange {
            OpsetRange {
                min: 2,
                max: Some(4),
            }
        }

        fn infer_output_types(&self, _node: &CustomNode) -> Result<Vec<ArgType>, ProcessError> {
            Ok(vec![ArgType::Tensor(TensorType::new(DType::F32, 2, None))])
        }

        fn forward(
            &self,
            _node: &CustomNode,
            _ctx: &mut CodegenContext<'_, '_>,
        ) -> Result<TokenStream, ProcessError> {
            Ok(TokenStream::new())
        }
    }

    fn registry_with_test_op() -> HookRegistry {
        let mut registry = HookRegistry::default();
        registry.add_custom_op(Box::new(TestOp));
        registry
    }

    #[test]
    fn coverage_checks_identity_and_opset_range() {
        let registry = registry_with_test_op();
        let mismatch = HookCoverage::Missing(MissingReason::OpsetMismatch {
            supported: OpsetRange {
                min: 2,
                max: Some(4),
            },
        });
        assert_eq!(
            registry.coverage("FftReal", "custom_domain", 3),
            HookCoverage::Covered
        );
        assert_eq!(registry.coverage("FftReal", "custom_domain", 1), mismatch);
        assert_eq!(registry.coverage("FftReal", "custom_domain", 5), mismatch);
        // Same op_type, different domain: distinct ONNX identity
        assert_eq!(
            registry.coverage("FftReal", "other_domain", 3),
            HookCoverage::Missing(MissingReason::NoHook)
        );
    }

    #[test]
    #[should_panic(expected = "Duplicate custom op registration")]
    fn duplicate_registration_panics() {
        let mut registry = registry_with_test_op();
        registry.add_custom_op(Box::new(TestOp));
    }

    /// Custom op in the DEFAULT ONNX domain (an unknown op_type there is a
    /// legitimate custom op), where the domain string is empty.
    struct DefaultDomainOp;

    impl CustomOp for DefaultDomainOp {
        fn op_type(&self) -> &str {
            "MyUnknownOp"
        }

        fn infer_output_types(&self, _node: &CustomNode) -> Result<Vec<ArgType>, ProcessError> {
            Ok(vec![ArgType::Tensor(TensorType::new(DType::F32, 2, None))])
        }

        fn forward(
            &self,
            _node: &CustomNode,
            _ctx: &mut CodegenContext<'_, '_>,
        ) -> Result<TokenStream, ProcessError> {
            Ok(TokenStream::new())
        }
    }

    #[test]
    #[should_panic(expected = "Duplicate custom op registration for 'MyUnknownOp'")]
    fn duplicate_default_domain_registration_names_the_op_without_a_bare_separator() {
        // The default domain is the empty string, so a naive
        // "{domain}::{op_type}" would read "::MyUnknownOp".
        let mut registry = HookRegistry::default();
        registry.add_custom_op(Box::new(DefaultDomainOp));
        registry.add_custom_op(Box::new(DefaultDomainOp));
    }

    /// Same op_type as `DefaultDomainOp`, declaring the default domain by its
    /// other spec-legal spelling.
    struct AiOnnxDomainOp;

    impl CustomOp for AiOnnxDomainOp {
        fn op_type(&self) -> &str {
            "MyUnknownOp"
        }

        fn domain(&self) -> &str {
            "ai.onnx"
        }

        fn infer_output_types(&self, _node: &CustomNode) -> Result<Vec<ArgType>, ProcessError> {
            Ok(vec![ArgType::Tensor(TensorType::new(DType::F32, 2, None))])
        }

        fn forward(
            &self,
            _node: &CustomNode,
            _ctx: &mut CodegenContext<'_, '_>,
        ) -> Result<TokenStream, ProcessError> {
            Ok(TokenStream::new())
        }
    }

    #[test]
    fn default_domain_spellings_match_each_other() {
        // A hook declaring "" must cover a node the model spells "ai.onnx",
        // and a hook declaring "ai.onnx" must cover a node spelled "".
        let mut registry = HookRegistry::default();
        registry.add_custom_op(Box::new(DefaultDomainOp));
        assert_eq!(
            registry.coverage("MyUnknownOp", "ai.onnx", 1),
            HookCoverage::Covered
        );

        let mut registry = HookRegistry::default();
        registry.add_custom_op(Box::new(AiOnnxDomainOp));
        assert_eq!(
            registry.coverage("MyUnknownOp", "", 1),
            HookCoverage::Covered
        );

        // Still a distinct identity from a genuinely different domain.
        assert_eq!(
            registry.coverage("MyUnknownOp", "ai.onnx.ml", 1),
            HookCoverage::Missing(MissingReason::NoHook)
        );
    }

    #[test]
    #[should_panic(expected = "Duplicate custom op registration for 'MyUnknownOp'")]
    fn registering_both_default_domain_spellings_is_a_duplicate() {
        // Without canonicalization these are two keys, and only whichever one
        // matches the model's spelling would ever fire.
        let mut registry = HookRegistry::default();
        registry.add_custom_op(Box::new(DefaultDomainOp));
        registry.add_custom_op(Box::new(AiOnnxDomainOp));
    }

    struct TestOverride(NodeType);

    impl OpOverride for TestOverride {
        fn target(&self) -> NodeType {
            self.0.clone()
        }

        fn forward(
            &self,
            _node: &Node,
            _ctx: &mut CodegenContext<'_, '_>,
        ) -> Result<TokenStream, ProcessError> {
            Ok(TokenStream::new())
        }
    }

    #[test]
    fn override_lookup_by_node_type() {
        let mut registry = HookRegistry::default();
        registry.add_override(Box::new(TestOverride(NodeType::MatMul)));
        assert!(registry.override_for(&NodeType::MatMul).is_some());
        assert!(registry.override_for(&NodeType::Relu).is_none());
    }

    #[test]
    #[should_panic(expected = "Duplicate op override registration")]
    fn duplicate_override_panics() {
        let mut registry = HookRegistry::default();
        registry.add_override(Box::new(TestOverride(NodeType::MatMul)));
        registry.add_override(Box::new(TestOverride(NodeType::MatMul)));
    }

    #[test]
    #[should_panic(expected = "cannot target NodeType::Custom")]
    fn override_targeting_custom_panics() {
        let mut registry = HookRegistry::default();
        registry.add_override(Box::new(TestOverride(NodeType::Custom)));
    }
}
