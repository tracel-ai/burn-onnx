# Custom Op Hooks for burn-onnx

Design document for [issue #23](https://github.com/tracel-ai/burn-onnx/issues/23):
"Add a way to implement custom function for operators not supported in ONNX format".

Status: implemented (branch `custom-op-hooks`, design v2 plus the revisions
below). This document is the design record: the rationale, the alternatives
considered, and where the landed code deviates ("Implementation
note/correction" blocks). **The maintained user-facing reference - worked
examples, type mappings, iteration workflow - is the "Custom Operators and
Overrides" section of `DEVELOPMENT-GUIDE.md`; when this document and that one
disagree, that one is right.** File:line references are survey notes against
the pre-implementation tree at `846b2452`; the implementation moved many of
them.

Post-implementation review revisions (2026-08-12):

- Trait methods that produce data (`forward`, `field`, `collect_snapshots` on
  both `CustomOp` and `OpOverride`) return `Result<_, ProcessError>` so a hook
  can reject a configuration it cannot handle. Code generation itself has no
  recoverable error channel (`BurnGraph::codegen` returns a `TokenStream`), so
  an `Err` fails the build with a message naming the node and the method; the
  gain over a panic inside the hook is attribution, not recoverability.
- `opset_range()` returns an `OpsetRange { min, max }` struct (shared with the
  coverage diagnostics) instead of a bare tuple.
- `HookCoverage` is `Covered | Missing(MissingReason)`; `MissingHook.reason`
  carries `MissingReason`, which makes a "missing but covered" state
  unrepresentable. `coverage() == Covered` followed by `infer() == Ok(None)`
  is enforced as an error (contract violation), not a silent fallback.
- `CustomNode` is `#[non_exhaustive]` with a `new()` constructor;
  `HookCoverage`, `MissingReason`, `MissingHook`, `AttrKind` are
  `#[non_exhaustive]`.
- Attribute surface: graph-valued attributes stay visible as payload-free
  markers, and `PublicAttributesOwned::kind()` distinguishes "absent" from
  "present with a different type".
- The hook-free inference fallback preserves types the model declared via
  `value_info` and fills every still-undeclared output, instead of
  unconditionally overwriting `outputs[0]`.
- `BurnGraph` rejects custom ops and override targets inside If/Loop/Scan
  bodies up front (subgraph body codegen is hook-free by design in v1), so
  the field/snapshot collection and forward codegen halves cannot disagree.

## 1. Goals

- Hook unrecognized ONNX ops (e.g. `custom_domain::FftReal`) to user-supplied Rust
  code instead of failing the build.
- Replace the codegen for any built-in op (e.g. swap every `MatMul` for a user's
  specialized kernel).
- Trait + builder API: users implement a `CustomOp` or `OpOverride` trait and
  register instances via `ModelGen` builder methods.
- Global scope: overrides apply to every node of the matched op type.

## 2. Non-goals (for v1)

- Per-node-name surgical overrides (deferred; the trait shape leaves room).
- User-extensible simplification passes (constant folding of custom ops).
- Subgraph (`If` / `Loop` / `Scan`) custom hooks. The design works for top-level
  nodes; subgraph integration is additive.

## 3. Architecture

```
┌───────────────────────────────────────────────────────────────┐
│ User's build.rs / CLI                                         │
│   ModelGen::new()                                             │
│     .register_custom_op(MyFft)                                │
│     .register_op_override(MyMatMul)                           │
│     .run_from_script();                                       │
└───────────────────────────┬───────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ burn-onnx (codegen layer) │
              │   ext: CustomOp/OpOverride│
              │        CodegenContext     │
              │   HookRegistry (codegen)  │
              └─────────────┬─────────────┘
                            │ HookRegistry implements
                            │ CustomOpInference; shared as
                            │ Arc<dyn CustomOpInference>
              ┌─────────────▼─────────────┐
              │ onnx-ir (IR + parse)      │
              │   NodeType::Custom        │
              │   Node::Custom(CustomNode)│
              │   CustomProcessor (global)│
              │   PipelineHooks (type     │
              │     inference only)       │
              └───────────────────────────┘
```

The user-facing traits live in `burn-onnx`'s `ext` module (they reference codegen
types via the `CodegenContext` / `Imports` wrappers). Their type-inference half
reaches `onnx-ir` as an `Arc<dyn CustomOpInference>` (the `HookRegistry` itself
implements that trait), consulted only during type inference (4.5), so user code
never depends on `onnx-ir` internals directly.

Why this split:

- `onnx-ir` stays pure. It exposes a narrow `CustomOpInference` trait that does
  not pull in `proc_macro2`, `quote`, or any burn types. Other consumers of
  `onnx-ir` are unaffected.
- `burn-onnx` owns the user-facing traits and the `ext` surface, and bridges the
  two layers.

## 4. Changes in `onnx-ir`

### 4.1 New `Custom` variants in `NodeType` and `Node`

`define_node_enum!` (`ir/node.rs:68`) is extended to allow a sentinel
`@custom Custom => custom::CustomNode` entry. The macro emits a unit
`NodeType::Custom` (so it can be used as a registry key; `NodeType` already
derives `Hash, Eq`) and a tuple `Node::Custom(CustomNode)`.

Two macro details that v1 missed:

- `NodeType` derives `EnumString` with `#[strum(ascii_case_insensitive)]`
  (`ir/node.rs:81`). Without protection, `NodeType::from_str("Custom")` (or
  `"custom"`) would match the new variant, so a default-domain ONNX op literally
  named `Custom` would skip the unknown-op path and alias the sentinel. The
  `@custom` arm must emit the variant with `#[strum(disabled)]`, which excludes
  it from `FromStr`. Implementation note: `NodeType`'s `Display` is hand-written
  in the macro rather than strum-derived, because the disabled variant must
  remain printable (node renaming and logging format `node_type`).
- The dispatch in 5.3 needs to look up overrides by `NodeType`. The macro should
  also emit a `Node::node_type(&self) -> NodeType` accessor (trivial match,
  alongside the existing generated `name()` / `inputs()` / `outputs()`).

```rust
// crates/onnx-ir/src/node/custom.rs (new)
#[derive(Debug, Clone)]
pub struct CustomNode {
    pub name: String,
    pub op_type: String,            // raw ONNX op_type, e.g. "FftReal"
    pub domain: String,             // raw ONNX domain, e.g. "custom_domain" or ""
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub attrs: PublicAttributesOwned,
    pub opset: usize,               // opset for THIS node's domain (see 4.2)
}
```

`Node::Custom`'s generated accessor methods (`name()`, `inputs()`, `outputs()`)
work unchanged because `CustomNode` has the same field names the macro expects.
No `config: ()` field. Custom ops do not have a typed config because their
schema is not known at `burn-onnx` compile time.

Because `inputs` are full `Argument` values (cloned with their `value_store`
attached), hooks can read constant input data via the already-public
`Argument::value() -> Option<TensorData>` (`ir/argument.rs:369`). See 4.6.

### 4.2 Parser fallback (domain-aware)

ONNX operator identity is the triple `(domain, op_type, opset-for-that-domain)`,
not `op_type` alone. `NodeProto` carries both `op_type` (field 4) and `domain`
(field 7), and `opset_import` is a list of `(domain, version)` pairs. Today the
domain is ignored entirely (the only `.domain` read in the crate is the
default-domain filter in `extract_opset_version`, `pipeline.rs:413`). A naive
`from_str(op_type)` fallback would resolve `custom_domain::MatMul` to the
built-in `MatMul`, silently bypassing the user's hook and applying default-domain
semantics. The fallback must therefore gate on the domain.

`proto_conversion.rs:607` changes from:

```rust
let node_type = NodeType::from_str(&node.op_type).expect("Unknown node type");
```

to a domain-aware resolution:

```rust
// Standard ONNX domains whose op_types map to built-in NodeTypes.
fn is_standard_domain(domain: &str) -> bool {
    matches!(domain, "" | "ai.onnx" | "ai.onnx.ml")
}

let node_type = if is_standard_domain(&node.domain) {
    NodeType::from_str(&node.op_type).unwrap_or(NodeType::Custom)
} else {
    // Any non-standard domain is always Custom, even if op_type collides
    // with a built-in name (e.g. custom_domain::MatMul stays Custom).
    NodeType::Custom
};
```

(With `#[strum(disabled)]` on the variant, `from_str` can never return `Custom`
by string match; the only way to become `Custom` is through this fallback.)

`RawNode` (`ir/node.rs:42`, currently `node_type` / `name` / `inputs` /
`outputs` / `attrs`) gains one optional field (implementation correction to v2,
which proposed three unconditional fields):

```rust
pub(crate) struct RawNode {
    // ... existing fields ...
    /// Raw (op_type, domain, domain opset) for custom ops; `None` for built-ins.
    pub custom_identity: Option<CustomIdentity>,
}

pub(crate) struct CustomIdentity {
    pub op_type: String,
    pub domain: String,
    pub domain_opset: usize,
}
```

Why `Option<CustomIdentity>` instead of unconditional fields: `RawNode` is
constructed literally at ~17 sites, many of them *synthetic* nodes created by
simplify passes (`coalesce_attention`, `constant_fold`, ...) that never came
from a proto and have no meaningful raw op_type or domain opset. "Populated
unconditionally" had no sensible value at those sites; `None` does, and the
invariant becomes checkable: `node_type == Custom` iff `custom_identity` is
`Some`.

#### Per-domain opset

`extract_opset_versions` (`pipeline.rs`) returns the model-level opset plus a
`HashMap<String, usize>` of every `(domain, version)` in `opset_import`.
`RawNode::domain_opset` is set from this map keyed by `raw_domain`. Per the ONNX
spec, every domain a node uses must appear in `opset_import`; for robustness
against malformed exporters we fall back to the default-domain opset (with a
`log::warn!`) rather than failing. This domain-specific opset is the value
checked against the hook's `opset_range()` gate in the coverage pre-pass (5.4)
and exposed on `CustomNode`, so a hook for `custom_domain` opset 2 sees `2`, not
the default ONNX opset.

`""` and `"ai.onnx"` are the same domain per the ONNX spec, so both the map keys
and a node's `CustomIdentity.domain` are canonicalized to `""` (and burn-onnx
canonicalizes a hook's declared `domain()` to match). A model that declares no
default-domain opset at all is not an error when it uses no default-domain
operators, which is the case for `ai.onnx.ml`-only exports;
`Error::MissingOpsetVersion` is raised only when default-domain nodes are
actually present.

### 4.3 Public `AttributeValue` surface

`AttributeValue` is `pub(crate)` (`ir/attribute.rs:106`) and `Attributes` is a
plain `pub type Attributes = HashMap<String, AttributeValue>` alias. Custom ops
need to read attributes (e.g. `n_fft`, `hop_length` for FFT), so we expose a
read-only wrapper rather than the internal enum:

```rust
// crates/onnx-ir/src/ir/attribute.rs
pub struct PublicAttributesOwned(/* private scalar/tensor value map */);

impl PublicAttributesOwned {
    pub(crate) fn from_internal(attrs: &Attributes) -> Self;
    pub fn get_i64(&self, name: &str) -> Option<i64>;
    pub fn get_i64s(&self, name: &str) -> Option<&[i64]>;
    pub fn get_f32(&self, name: &str) -> Option<f32>;
    pub fn get_f32s(&self, name: &str) -> Option<&[f32]>;
    pub fn get_string(&self, name: &str) -> Option<&str>;
    pub fn get_strings(&self, name: &str) -> Option<&[String]>;
    pub fn get_tensor(&self, name: &str) -> Option<&TensorData>;
    pub fn get_tensors(&self, name: &str) -> Option<&[TensorData]>;
    pub fn names(&self) -> impl Iterator<Item = &str>;
}
```

Reason for the wrapper: the internal enum has `DeferredGraph` / `Graph` variants
and `pub(crate)` access we don't want to leak (graph-valued attributes are
explicitly out of scope for v1, matching non-goal "subgraph hooks"). Hiding it
behind getter methods keeps the existing internals private. The owned form is
attached to `CustomNode` so the user can hold the attributes past the parse
scope.

Implementation correction: an earlier draft also had a borrowed
`PublicAttributes<'a>(&'a Attributes)` view. It was dropped: in the v2 contract
every hook method receives `&CustomNode` (or the typed `Node`), so the borrowed
form has no consumer, and `CustomNode` must own its attributes anyway because
it outlives the parse.

Implementation correction: the owned form must NOT wrap the internal
`AttributeValue` map. Auto traits are type-based, so embedding `AttributeValue`
(whose `DeferredGraph` variant holds an `Rc`-backed `NameRegistry`) in
`CustomNode` makes `Node` and `OnnxGraph` lose `Send`/`Sync`, breaking
downstream users (e.g. `OnceLock<OnnxGraph>` test fixtures).
`PublicAttributesOwned` therefore stores its own private value enum with only
scalar/tensor variants, converted (and graph attributes dropped) at snapshot
time.

### 4.4 `CustomProcessor` (hook-free, globally registered)

This is the structural correction from v1. The registry is consulted in four
places, and the most important one for `Custom` nodes cannot take a threaded
hooks parameter:

`build_node` is dispatched from `convert_builders_to_nodes`
(`ir/graph.rs:188-205`), a free function called by the public
`OnnxGraphBuilder::convert_to_graph(self, opset)` (`ir/graph.rs:61`), which is
in turn called from `pipeline.rs:289` and from
`DeferredGraph::build_graph_with_outer_scope` (`ir/attribute.rs:85,100`).
Threading a hooks value through `convert_to_graph` would change a public
signature and still not cover all entry points.

The fix follows from an observation: every `NodeProcessor` responsibility for a
custom node is hook-independent except `infer_types`:

| Method | Custom-node behavior | Needs the user hook? |
|---|---|---|
| `spec()` | permissive `NodeSpec::default()` (min_opset 1, any I/O count) | no |
| `lift_constants()` | no-op | no |
| `input_preferences()` | `None` (v1; see open question 8) | no |
| `infer_types()` | call the user's `infer_output_types` | **yes** |
| `is_noop()` | `false` | no |
| `build_node()` | build the `Node::Custom(CustomNode)` view | no |

So a hook-free `CustomProcessor` is registered for `NodeType::Custom` in the
global registry (`ProcessorRegistry::with_standard_processors`,
`registry.rs`), exactly like every built-in processor. `build_node` then works
everywhere `convert_to_graph` is reached, with zero plumbing changes. Only type
inference needs the hook, and that is handled by the overlay in 4.5.

```rust
// crates/onnx-ir/src/node/custom.rs

/// Build the public CustomNode view from a RawNode at any pipeline stage.
fn custom_node_view(node: &RawNode) -> CustomNode {
    CustomNode {
        name:    node.name.clone(),
        op_type: node.raw_op_type.clone(),
        domain:  node.raw_domain.clone(),
        inputs:  node.inputs.clone(),
        outputs: node.outputs.clone(),
        attrs:   PublicAttributesOwned::from_internal(&node.attrs),
        opset:   node.domain_opset,
    }
}

pub(crate) struct CustomProcessor;

impl NodeProcessor for CustomProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec { NodeSpec::default() }   // permissive

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Reached only when inference runs without hooks (a direct onnx-ir
        // consumer, or post-pre-pass invariant violation). Best-effort
        // fallback keeps the graph buildable for inspection/debugging.
        // Guard the indexing: same_as_input() indexes inputs[0]/outputs[0].
        if !node.inputs.is_empty() && !node.outputs.is_empty() {
            same_as_input(node);
        }
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Custom(custom_node_view(&builder))
    }
}
```

The hook-aware path wraps this with the user inference:

```rust
pub(crate) struct HookedCustomProcessor {
    hooks: Arc<dyn CustomOpInference>,
}

impl NodeProcessor for HookedCustomProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec { NodeSpec::default() }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let view = custom_node_view(node);
        match self.hooks.infer(&view)? {
            Some(types) => {
                if types.len() != node.outputs.len() {
                    return Err(ProcessError::Custom(format!(
                        "Custom op '{}' ({}::{}) returned {} output type(s) \
                         but the node has {} output(s)",
                        node.name, node.raw_domain, node.raw_op_type,
                        types.len(), node.outputs.len(),
                    )));
                }
                for (out, ty) in node.outputs.iter_mut().zip(types) {
                    out.ty = ty;
                }
                Ok(())
            }
            // No hook for this (op_type, domain): the coverage pre-pass (5.4)
            // already failed before we got here when hooks are in play, so
            // this mirrors the hook-free fallback.
            None => CustomProcessor.infer_types(node, _opset, _output_preferences),
        }
    }
}
```

`OutputPreferences` are ignored for custom ops in v1: consumers may request
`Shape` / `ScalarNative` outputs, but the hook is the sole authority on its
output types. Documented on the trait.

### 4.5 Threading the hook through type inference

Verified registry consultation map (the v1 doc had `build_node` in the wrong
phase):

| Site | Methods used | Hook needed? |
|---|---|---|
| `phases/node_conversion.rs:194` | `lift_constants` | no (no-op for Custom) |
| `phases/type_inference.rs:39,117,187` | spec validation, `infer_types`, `input_preferences` | **yes** (`infer_types`) |
| `phases/post_processing.rs:58-67,260-265` | `is_noop`, re-lift `lift_constants` | no |
| `ir/graph.rs:188-205` (via `convert_to_graph`) | `build_node` | no (global `CustomProcessor`, 4.4) |

With the hook-free half registered globally, only the type-inference phase needs
an overlay. `PipelineHooks` owns one `HookedCustomProcessor` (built once per
parse from the registered inference hook):

```rust
pub(crate) struct PipelineHooks {
    custom: HookedCustomProcessor,   // wraps Arc<dyn CustomOpInference>
}

impl PipelineHooks {
    /// Resolution point used by the type-inference phase in place of
    /// `registry.get(node_type)`.
    fn resolve<'a>(
        &'a self,
        node_type: &NodeType,
        registry: &'a ProcessorRegistry,
    ) -> &'a dyn ProcessorMethods {
        match node_type {
            NodeType::Custom => &self.custom,
            other => registry.get(other),
        }
    }
}
```

The lifetimes work because `PipelineHooks` outlives the phase calls (created in
`build_graph_builder_from_proto_with_outer_scope`, borrowed by the phase), so
`resolve` can hand back a reference into either the global registry or the
long-lived processor. (`HookedCustomProcessor` gets `ProcessorMethods` for free
through the existing blanket impl, `registry.rs:32`.)

Concrete plumbing changes:

1. `build_graph_builder_from_proto_with_outer_scope(...)` (`pipeline.rs:329`)
   takes `hooks: &PipelineHooks` and forwards it to the type-inference phase and
   to node conversion (the latter only so DeferredGraph creation can capture the
   `Arc`, next point). `simplify: bool` already rides through these signatures
   the same way, so there is precedent.
2. The type-inference phase replaces its three `registry.get(...)` calls with
   `hooks.resolve(...)`. For `input_preferences` (line 187) the custom processor
   returns the default `None`; routing it through `resolve` anyway keeps a
   single resolution rule.
3. `DeferredGraph` (`ir/attribute.rs:42`) gains
   `custom_op_inference: Option<Arc<dyn CustomOpInference>>` (cheap clone).
   DeferredGraphs are created during proto conversion
   (`proto_conversion.rs:838,851`) and consumed by the If/Loop/Scan processors
   during their `infer_types` (`if_node.rs:174,195`, `loop_node.rs:297`,
   `scan_node.rs:180`), which call `build_graph_with_outer_scope` and so re-enter
   the pipeline. Storing the `Arc` at creation is the only way the hook reaches
   subgraph builds, which run inside the parent's type-inference phase with no
   other channel back to the entry point.
4. The public entry point seeds it all. Note the crate has two types named
   `OnnxGraphBuilder`: the public parse API (`pipeline.rs:126`, exported from
   `lib.rs`) and the internal intermediate graph (`ir/graph.rs:42`). The change
   is to the public one, matching its existing consuming-builder style
   (`simplify(mut self, ...) -> Self`):

```rust
// pipeline.rs
pub struct OnnxGraphBuilder {
    simplify: bool,
    custom_op_inference: Option<Arc<dyn CustomOpInference>>,
}

impl OnnxGraphBuilder {
    pub fn with_custom_op_inference(mut self, h: Arc<dyn CustomOpInference>) -> Self {
        self.custom_op_inference = Some(h);
        self
    }
}
```

When `custom_op_inference` is `None`, `PipelineHooks` wraps a no-op inference
that always returns `Ok(None)` / `Missing(NoHook)`, so behavior for models with no custom
ops is unchanged. The global singleton registry is never mutated after init, and
nothing new is allocated per build beyond the single processor.

Implementation note: rather than a separate no-op inference impl,
`HookedCustomProcessor` holds `Option<Arc<dyn CustomOpInference>>` and treats
`None` identically to a hook returning `Ok(None)` (fall through to the
hook-free fallback). Same behavior, one less type.

#### The `CustomOpInference` trait

A narrow, object-safe interface defined in `onnx-ir` and implemented by
`burn-onnx`'s `HookRegistry`. It takes the public `CustomNode` view (4.6), never
the internal `RawNode`:

```rust
/// Inclusive opset range a hook supports (also used by CustomOp::opset_range).
pub struct OpsetRange { pub min: usize, pub max: Option<usize> }  // + contains()

/// Why a custom op is not covered.
#[non_exhaustive]
pub enum MissingReason {
    NoHook,
    OpsetMismatch { supported: OpsetRange },
}

/// Coverage answer for the pre-pass (5.4).
#[non_exhaustive]
pub enum HookCoverage {
    Covered,
    Missing(MissingReason),
}

pub trait CustomOpInference: Send + Sync {
    /// Coverage for this (op_type, domain) at the node's domain opset.
    /// Used by the coverage pre-pass before type inference runs.
    fn coverage(&self, op_type: &str, domain: &str, opset: usize) -> HookCoverage;

    /// Infer output types. `Ok(None)` => no hook is registered for this node.
    /// May be called more than once per node (fixed-point inference loop).
    fn infer(&self, node: &CustomNode) -> Result<Option<Vec<ArgType>>, ProcessError>;
}
```

The `burn-onnx` implementation dispatches to the matching user
`CustomOp::infer_output_types`, after checking `opset_range()`. The
`Covered | Missing(reason)` shape (a review revision from the flat
`Covered / NoHook / OpsetMismatch`) makes a "missing but covered" diagnostic
unrepresentable, and the contract is enforced: because the coverage pre-pass
runs on the whole graph before inference, a hook answering `Covered` and then
returning `Ok(None)` from `infer` is reported as a contract-violation error,
never silently substituted with the fallback guess.

### 4.6 Public extension surface (onnx-ir side)

Several types the trait contract depends on are currently unreachable from user
crates. This must be fixed explicitly, or the traits cannot be implemented.
Verified current state:

| Type | Current state | Needed |
|---|---|---|
| `NodeType` | declared `pub`, but only re-exported `pub(crate)` (`ir/mod.rs:16`) | `pub` re-export |
| `ProcessError` | `pub` enum inside private `mod processor` (`lib.rs:10`) | `pub` re-export |
| `CustomNode` | new | `pub` (via `pub mod node`) |
| `ArgType`, `Argument`, `TensorType`, `TensorData`, `DType` | already `pub` via `pub use ir::*` | unchanged |
| `Argument::value()` | already `pub` (`ir/argument.rs:369`) | unchanged; document |
| `RawNode` | `pub(crate)` | stays private (never in public API) |

```rust
// crates/onnx-ir/src/lib.rs
pub use ir::NodeType;                       // promote the re-export
pub use processor::ProcessError;            // expose the error type
pub use node::custom::CustomNode;           // plus HookCoverage, CustomOpInference
// PublicAttributesOwned rides the existing `pub use ir::*;`
```

`RawNode` deliberately stays crate-private; the public contract is expressed
entirely through `CustomNode` and `PublicAttributesOwned`, which is why
`CustomOpInference::infer` takes `&CustomNode` rather than `&RawNode`.

Constant inputs deserve a callout because they make ops like FFT practical:
value stores are attached to arguments during node conversion
(`phases/node_conversion.rs:165`), so by the time `infer_output_types` or
`forward` sees a `CustomNode`, `node.inputs[i].value()` returns the
`TensorData` of any constant/initializer input (`None` for dynamic ones). No
new API is needed; the docs for `CustomNode` must state this.

Making `NodeType` public also lets users name `NodeType::MatMul` for
`OpOverride::target()` (5.1). `ProcessError`'s variants are already
constructible (plain enum fields), so user hooks can return
`ProcessError::MissingAttribute(...)` etc. One consequence to accept: promoting
these re-exports makes them semver-relevant for `onnx-ir`.

### 4.7 Custom nodes are opaque to simplification

The simplify passes (CSE, constant folding, pattern rewrites under
`crates/onnx-ir/src/simplify/`) reason about node semantics. A custom op's
semantics are unknown at parse time, and may be stateful or random (the
ONNX contrib world contains both). Rule for v1, enforced in whatever pass
enumerates candidates:

- Never CSE-merge two `Custom` nodes, even with identical inputs/attrs.
- Never constant-fold through a `Custom` node.
- Pattern rewrites must not match across a `Custom` node.

This is a small, easily-reviewed exclusion (`node_type == NodeType::Custom`
checks at candidate-collection sites) and is required for correctness, not an
optimization choice.

Audit result (implementation): only CSE (`simplify/redundant_nodes.rs`) selects
candidates generically by `(type, inputs, attrs)` key and needed an explicit
exclusion. All other passes (`constant_fold`, `constant_shape`,
`identity_element`, `idempotent`, `permute_reshape`, `coalesce_attention`,
no-op elimination via `is_noop`) match explicit built-in `NodeType`s and are
naturally safe. Dead-node elimination intentionally still applies to custom
nodes: ONNX graphs are functional, and removing an unused output is standard
behavior (onnxruntime does the same).

## 5. Changes in `burn-onnx`

### 5.0 Public extension surface (burn-onnx side)

The codegen-facing types are `pub` structs trapped in private modules
(`mod scope`, `mod imports`, `mod node_traits`; see `burn/mod.rs`), so they are
not reachable either. Rather than blanket-exposing those modules, introduce a
single curated `burn_onnx::ext` module:

```rust
// crates/burn-onnx/src/ext.rs (new) -- the ONLY entry point users import
pub use crate::burn::custom_op::{CustomOp, OpOverride};
pub use crate::burn::node_traits::{arg_to_ident, create_lazy_snapshot, Field};

// Re-export the onnx-ir types users need so they need not depend on the
// exact onnx-ir version directly.
pub use onnx_ir::{
    ArgType, Argument, CustomNode, DType, Node, NodeType, ProcessError,
    TensorData, TensorType,
};

// Snapshot type used by collect_snapshots (from burn-store).
pub use burn_store::TensorSnapshot;

// Token-stream crates. TokenStream values must come from the same proc-macro2
// build burn-onnx links; cargo unifies 1.x in practice, but re-exporting
// removes the failure mode entirely and saves users two dependencies.
pub use proc_macro2;
pub use quote;

/// Public, stable wrapper passed to `forward()`. Hides ScopeAtPosition behind
/// the one operation a hook needs.
pub struct CodegenContext<'a, 'b> {
    inner: &'a mut crate::burn::scope::ScopeAtPosition<'b>,
}

impl<'a, 'b> CodegenContext<'a, 'b> {
    /// Resolve an input argument to a token stream (handles clone tracking,
    /// Tensor/Scalar/Shape, exactly like the built-in nodes' `scope.arg()`).
    pub fn arg(&mut self, arg: &Argument) -> proc_macro2::TokenStream {
        self.inner.arg(arg)
    }
}

/// Public wrapper over the internal BurnImports, passed to `register_imports`.
pub struct Imports<'a> {
    inner: &'a mut crate::burn::imports::BurnImports,
}

impl<'a> Imports<'a> {
    /// Register an import path, e.g. `use crate::ops;` (resolved inside the
    /// user's crate, where the generated file is included).
    pub fn register(&mut self, path: impl Into<String>) {
        self.inner.register(path);
    }
}
```

Corrections to v1 in this list: `arg_to_ident` lives in `node_traits.rs:168`
(not `argument_helpers.rs`) and is currently not re-exported anywhere, so the
`ext` re-export is its promotion point. `create_lazy_snapshot`
(`node_traits.rs:197`) is what built-in nodes use to build `TensorSnapshot`s
from constant inputs; custom ops with weights need it for `collect_snapshots`,
so it is part of the curated surface.

Why wrappers instead of exposing `ScopeAtPosition` / `BurnImports`: the honest
rationale (v1 overstated it) is not that `ScopeAtPosition`'s own surface is
broad; it has exactly three public methods (`arg`, `scope`, `node_position`).
The problem is that `scope()` returns `&mut Scope`, whose clone-tracking and
partitioning API would become frozen public API transitively. `CodegenContext`
exposes only `arg`, and `Imports` only `register`, which is everything the
built-in nodes' `forward()` / `register_imports()` bodies use. Tradeoff: a thin
indirection layer to maintain. The alternative (make the modules public) is
recorded in the open questions. Import registration stays in the dedicated
`register_imports` method (mirroring `NodeCodegen`) so there is exactly one
place to add imports and the ordering matches the built-in import-collection
pass.

### 5.1 The `CustomOp` trait

Changed from v1: methods take `&self`. This makes the traits object-safe, so
`HookRegistry` stores `Box<dyn CustomOp>` directly and the entire
`ErasedCustomOp` / `ErasedOpOverride` closure-bridging layer from v1 is deleted.
It also means hooks can carry state (paths, flags, precomputed tables), which
static methods forbid. `Send + Sync` is required because the registry is shared
into the onnx-ir pipeline as `Arc<dyn CustomOpInference>`.

All trait inputs and outputs come from the public `ext` surface (4.6, 5.0).

```rust
// crates/burn-onnx/src/burn/custom_op.rs (new)
use crate::ext::{
    ArgType, CodegenContext, CustomNode, Field, Imports, ProcessError, TensorSnapshot,
};
use proc_macro2::TokenStream;

pub trait CustomOp: Send + Sync + 'static {
    /// ONNX op_type this hook handles (e.g. "FftReal").
    fn op_type(&self) -> &str;

    /// ONNX domain. Empty string = default ONNX domain.
    fn domain(&self) -> &str { "" }

    /// Opset gate, checked against the node's domain opset by the coverage
    /// pre-pass (5.4). Out-of-range is reported as a coverage error.
    fn opset_range(&self) -> OpsetRange { OpsetRange::from_min(1) }

    /// Infer output ArgTypes. Called during onnx-ir type inference (possibly
    /// more than once per node; the loop is a fixed point). MUST return
    /// exactly `node.outputs.len()` types; the processor rejects a mismatch
    /// with a ProcessError (4.4). Constant inputs are readable via
    /// `node.inputs[i].value()`.
    fn infer_output_types(&self, node: &CustomNode) -> Result<Vec<ArgType>, ProcessError>;

    /// Generate the forward-pass code for this node. Err fails the build with
    /// a message naming the op and method (review revision: hooks previously
    /// returned a bare TokenStream and could only panic).
    fn forward(&self, node: &CustomNode, ctx: &mut CodegenContext<'_, '_>)
        -> Result<TokenStream, ProcessError>;

    /// Optional: extra imports.
    fn register_imports(&self, _imports: &mut Imports<'_>) {}

    /// Optional: declare a module field (e.g. learnable params or RNG state).
    fn field(&self, _node: &CustomNode) -> Result<Option<Field>, ProcessError> { Ok(None) }

    /// Optional: weights/snapshot collection (parallels NodeCodegen).
    fn collect_snapshots(&self, _node: &CustomNode, _field_name: &str)
        -> Result<Vec<TensorSnapshot>, ProcessError> { Ok(vec![]) }
}
```

`OpOverride` matches by `NodeType` instead of `(op_type, domain)` and receives
the typed `Node`. It exposes the same overridable codegen surface as `CustomOp`
(`forward`, `register_imports`, `field`, `collect_snapshots`) so the dispatch in
5.3 has a method to call for each. It deliberately has no `infer_output_types`:
the built-in processor already produced correct types, and overrides are
codegen-only (see open question 2).

```rust
use crate::ext::Node;

pub trait OpOverride: Send + Sync + 'static {
    fn target(&self) -> NodeType;

    fn forward(&self, node: &Node, ctx: &mut CodegenContext<'_, '_>)
        -> Result<TokenStream, ProcessError>;

    fn register_imports(&self, _imports: &mut Imports<'_>) {}
    fn field(&self, _node: &Node) -> Result<Option<Field>, ProcessError> { Ok(None) }
    fn collect_snapshots(&self, _node: &Node, _field_name: &str)
        -> Result<Vec<TensorSnapshot>, ProcessError> { Ok(vec![]) }
}
```

Because the existing processor already did type inference correctly, the override
does not touch `onnx-ir` at all.

### 5.2 `ModelGen` builder additions

Matches the existing `&mut self -> &mut Self` builder style
(`model_gen.rs:155-247`):

```rust
impl ModelGen {
    pub fn register_custom_op(&mut self, op: impl CustomOp) -> &mut Self {
        self.hooks.add_custom_op(Box::new(op));
        self
    }

    pub fn register_op_override(&mut self, ov: impl OpOverride) -> &mut Self {
        self.hooks.add_override(Box::new(ov));
        self
    }
}
```

`HookRegistry` (internal) stores both, keyed at registration time:

```rust
struct HookRegistry {
    customs:   HashMap<(String, String), Box<dyn CustomOp>>,  // (op_type, domain)
    overrides: HashMap<NodeType,         Box<dyn OpOverride>>,
}
```

Duplicate registration for the same key is a `panic!` in the builder (build.rs
context; immediate, attributable). `HookRegistry` (behind `Arc`) implements
`CustomOpInference` (4.5): `coverage` checks key presence then `opset_range`;
`infer` looks up and delegates to `infer_output_types`. `ModelGen::run` passes
the `Arc` to `OnnxGraphBuilder::with_custom_op_inference` at the existing parse
site (`model_gen.rs:346`).

Implementation note: `ModelGen` stores `Arc<HookRegistry>` directly;
`register_custom_op` mutates through `Arc::get_mut` (always succeeds during
builder setup, before any clone is handed out). The same `Arc` is cloned into
the parse pipeline and into `BurnGraph::with_hooks` for codegen.

### 5.3 Codegen dispatch with overrides

The `impl_node_codegen_dispatch!` macro (`node_codegen.rs:11`) already has
catch-all arms (`_ => panic!("Unsupported node type ...")` for
`inputs`/`outputs`/`forward`; silent defaults for the rest), so before PR 5
lands, a `Node::Custom` reaching codegen panics with a clear message; PR 1
upgrades those catch-all arms to name the custom op. The macro-generated impl is
renamed to `*_builtin` inherent methods, and the macro gains explicit
`Node::Custom(c)` arms for the structural accessors (`&c.inputs` / `&c.outputs`);
without them the builtin catch-all would panic on wiring queries that hooks
never influence.

Only the codegen-output methods consult hooks; the structural accessors
`inputs()` / `outputs()` never do, because neither a custom op nor an override
changes the graph's wiring.

The `NodeCodegen` methods gain a `&HookRegistry` parameter (the trait is `pub`
in the private `node_traits` module, so this is not a public-API break). Each
output method resolves override first, then `Custom` hook, then builtin.

```rust
impl NodeCodegen for Node {
    // Structural accessors: never routed through hooks.
    fn inputs(&self)  -> &[Argument] { self.inputs_builtin() }
    fn outputs(&self) -> &[Argument] { self.outputs_builtin() }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>, hooks: &HookRegistry)
        -> TokenStream
    {
        let mut ctx = CodegenContext::wrap(scope);
        if let Some(over) = hooks.override_for(&self.node_type()) {
            return over.forward(self, &mut ctx);
        }
        if let Node::Custom(c) = self {
            return hooks
                .custom_for(&c.op_type, &c.domain)
                .expect("checked by the coverage pre-pass; missing hook is a bug")
                .forward(c, &mut ctx);
        }
        self.forward_builtin(scope)
    }

    fn field(&self, hooks: &HookRegistry) -> Option<Field> {
        if let Some(over) = hooks.override_for(&self.node_type()) {
            return over.field(self);
        }
        if let Node::Custom(c) = self {
            return hooks.custom_for(&c.op_type, &c.domain).unwrap().field(c);
        }
        self.field_builtin()
    }

    // register_imports() and collect_snapshots() take `&HookRegistry` and follow
    // the same pattern: override first, then Custom hook, then builtin.
}
```

(`node_type()` is the new macro-emitted accessor from 4.1. The `expect` is an
internal invariant, not user-facing validation: `ModelGen` always runs the
coverage pre-pass before codegen, so a miss here is a burn-onnx bug.)

Threading: `BurnGraph` owns the `HookRegistry` (handed to it by `ModelGen`) and
passes `&HookRegistry` into every `NodeCodegen` call site. All of those live in
`graph.rs` (verified: `register_imports` at 475, `forward` at 688, `field` at
885, `collect_snapshots` at 997; the `inputs`/`outputs` reads at 738/741 are
structural and unchanged). `partition.rs` has no direct `NodeCodegen` calls, so
partitioned models pick up hooks for free through `graph.rs`.
`ScopeAtPosition` itself does not carry the registry; this keeps it out of the
scope's clone-tracking state.

Implementation correction: the hook-aware surface landed as free dispatch
functions in `node_codegen.rs` (`node_forward`, `node_field`,
`node_register_imports`, `node_collect_snapshots`), not as new parameters on
the `NodeCodegen` trait. Adding `&HookRegistry` to the trait would have
re-signatured every per-node impl (~150 files) for a parameter none of them
use; `Node` is a foreign type, so inherent methods were not an option either.
`graph.rs` call sites route through the free functions; the trait and all
per-node impls are untouched. Custom ops inside If/Loop/Scan subgraph bodies
still reach the trait's `Node::Custom` panic arm (codegen for those is the
known v1 punt, open question 1).

One behavior worth knowing: a custom op's constant input still goes through
the standard initializer-to-Constant path, producing a zero-initialized
`Param` field in the generated struct. A hook that reads the value via
`Argument::value()` and inlines it leaves that `Param` unused (harmless, and
`Model::new` stays safe); a hook that instead consumes the input via
`ctx.arg()` gets the runtime tensor, which is only correct when the model is
loaded with `from_file`/`from_bytes`.

### 5.4 Missing-hook validation (coverage pre-pass in onnx-ir)

Validation cannot wait until `ModelGen::run` inspects the finished graph. If a
custom op has no hook, the processor falls back to `same_as_input`, whose
guessed output type can be wrong, and a *downstream* node's type/shape validation
then fails during parse with an unrelated error before `ModelGen` ever receives a
graph. The friendly "missing hook" summary would never be reached.

So the coverage check runs as a dedicated pre-pass inside the pipeline,
immediately after Phase 2 (node conversion) and before Phase 3 (type inference):

```
PHASE 2  Node Conversion (Proto -> RawNode)
PHASE 2c Custom-op coverage check   <-- new, fails fast with full list
PHASE 3  Type Inference
```

The pass walks the `RawNode` list, and for every `node_type == NodeType::Custom`
asks `hooks.coverage(&raw_op_type, &raw_domain, domain_opset)`. It accumulates
all uncovered `(op_type, domain)` pairs with usage counts, carrying each
`MissingReason` (`NoHook`, or `OpsetMismatch` with the hook's supported range),
and if any exist returns `pipeline::Error::MissingCustomOpHooks(Vec<MissingHook>)`.
Because this runs before any type inference, it is never preempted by a cascade
error. Subgraph builds re-enter the pipeline (4.5 point 3), so a custom op
inside an `If` branch is checked when that branch is built; its error surfaces
through the parent's `ProcessError` path with the same payload.

Note: `pipeline::Error` is publicly exported (`lib.rs`), so the new variant is
technically a breaking change for exhaustive matches. Adding
`#[non_exhaustive]` to `Error` in the same release is recommended (it is an
error-reporting enum; downstream exhaustive matching is unlikely but possible).

`onnx-ir` only knows about inference hooks (it has the `CustomOpInference`
object), which is exactly the right scope: a custom op with no inference hook
cannot be type-inferred, so failing here is correct. Built-in op overrides live
only in `burn-onnx` and need no onnx-ir validation (the built-in always has a
processor).

Implementation note on gating: the pre-pass runs only when an inference hook
object is registered at all. A hook-less `OnnxGraphBuilder` parse keeps the
tolerant same-as-input fallback (the PR 1 debugging value: unknown-op models
stay inspectable). `ModelGen` therefore passes its `HookRegistry` Arc
unconditionally, even when empty, so build scripts always get the aggregated
missing-hook summary. The onnx-ir error text does not name `ModelGen`
(layering); `ModelGen` matches `Error::MissingCustomOpHooks` and appends the
"Register hooks via ModelGen::register_custom_op" hint itself.

`ModelGen` maps the error to a user-facing message:

```
error: ONNX file 'mamba.onnx' contains 3 custom op(s) with no registered hook:
  - mamba_domain::SelectiveScan   used by 12 node(s)
  - mamba_domain::CausalConv1d    used by 6 node(s)   (hook covers opsets 1..=1, model uses 3)
  - mamba_domain::RMSNormFused    used by 1 node(s)
Register hooks via ModelGen::register_custom_op(MyOp).
```

This replaces the current behavior of panicking at
`NodeType::from_str(...).expect("Unknown node type")` (`proto_conversion.rs:607`).

## 6. Generated code shape

Abridged real output from the `onnx-tests` custom-op fixture (three hooked
custom ops plus an overridden Relu). Two things to notice: the model is
monomorphic over the `burn::prelude` aliases (`Tensor<2>`, `Device`) - there
is no `B: Backend` parameter - and emitted paths resolve inside the *user's*
crate, because the generated file is `include!`-ed there (so same-crate
functions are `crate::...`, not a crate-name path):

```rust
// generated custom_ops.rs (abridged)
use burn::prelude::*;
use crate::custom_ops::ops;              // pushed by register_imports

impl Model {
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let custom1_out1 = ops::scale_shift(x, 2f32, 0.5f32);          // CustomOp
        let custom2_out1 = crate::custom_ops::ops::add_window(
            custom1_out1,
            &[0.25f32, 0.5f32, 0.75f32, 1f32],   // constant input inlined
            &self.device,                        // struct device is reachable
        );
        let custom3_out1 = custom2_out1;                               // CustomOp
        let relu1_out1 = crate::custom_ops::ops::my_relu(custom3_out1); // OpOverride
        relu1_out1
    }
}
```

The user controls the emitted path via their `forward()` impl: a free
function, a struct method, a trait method. The `ArgType` to generated-Rust
type mapping (`Tensor<N>` / `Tensor<N, Int>` / native scalars / `[i64; N]`
shapes) and the output-binding convention are documented in
`DEVELOPMENT-GUIDE.md`.

## 7. Example user code

Maintained in `DEVELOPMENT-GUIDE.md`, section "Custom Operators and
Overrides": a complete `CustomOp` (attributes, constant inputs, imports), an
`OpOverride`, the type-mapping table, the path-resolution rules, and the
discovery/iteration workflow. A second complete, *compiled* example lives in
`crates/onnx-tests/build.rs` (hooks) + `crates/onnx-tests/tests/custom_ops/`
(runtime ops and the e2e test). This section previously carried its own copy
of the example; it drifted from the landed API and was replaced by these
pointers.

## 8. Implementation sequence

Six PRs, each independently reviewable:

1. Parser tolerance + domain-aware fallback (4.2). Add `is_standard_domain` gate
   so non-standard domains always become `NodeType::Custom`. Introduce
   `NodeType::Custom` (`#[strum(disabled)]`) + `Node::Custom(CustomNode)` +
   `CustomNode` + the `node_type()` macro accessor +
   `raw_op_type` / `raw_domain` / `domain_opset` on `RawNode` + the per-domain
   opset map. Register the hook-free `CustomProcessor` in the global registry
   (4.4) so parse, `build_node`, and the simplify exclusions (4.7) work with no
   threading. Codegen gets explicit structural-accessor arms plus a clear
   "custom op requires a registered hook" panic in the dispatch catch-all arms.
   This alone lets users parse models with unknown ops, useful for debugging.

2. Public extension surface (4.6, 5.0). Promote the `NodeType` re-export,
   re-export `ProcessError`, add the `burn_onnx::ext` module with
   `CodegenContext`, `Imports`, and the curated re-exports (including
   `proc_macro2` / `quote`). No behavior change; pure surfacing.

3. `PublicAttributesOwned` accessor completeness. The owned form itself lands
   with PR 1 (`CustomNode` needs it); this PR adds tests on each accessor. The
   borrowed `PublicAttributes<'a>` from the earlier draft is dropped (4.3).

4. `CustomOp` trait + `CustomOpInference` + `PipelineHooks` threading (4.5,
   type-inference phase only) + `HookRegistry` + `ModelGen::register_custom_op`
   (custom ops only, no overrides yet). Generated code for a small set of
   integration-test ops (a no-op identity custom op, an FFT-like op, one with a
   constant input read via `Argument::value()`). Includes the `DeferredGraph`
   `Arc` plumbing.

5. `OpOverride` trait + dispatch wrapping in `node_codegen.rs` (5.3). Re-uses the
   same `HookRegistry`. Adds the `&HookRegistry` parameter to the internal
   `NodeCodegen` methods and threads it through the `graph.rs` call sites.

6. Coverage pre-pass (5.4) as Phase 2c in onnx-ir +
   `Error::MissingCustomOpHooks` (+ `#[non_exhaustive]` on `Error`) +
   `ModelGen` diagnostics. Doc updates in `DEVELOPMENT-GUIDE.md` and
   `SUPPORTED-ONNX-OPS.md` (a new "Extending burn-onnx" section).

PRs 1 and 6 can be sequenced first as a pair if the goal is "parse unknown ops
and fail with a friendly message" before any codegen capability lands.

## 9. Open questions

1. Subgraph custom ops (`If` / `Loop` / `Scan` bodies). The *inference* side now
   works through the `DeferredGraph` `Arc` (4.5), and the coverage pre-pass runs
   on subgraph builds too (5.4). The remaining v1 punt is codegen: the path that
   builds subgraph forward bodies needs the same dispatch wrap. Doable but
   additive; design supports it.

2. Should overrides be allowed to also affect type inference? Currently no.
   Overrides only change codegen, and the built-in processor still runs. If a
   user wants different output dtypes (e.g., a MatMul-int8 override), they'd
   need to use the `CustomOp` path with a non-empty domain and register a
   different op_type. V1 keeps override scope tight (codegen-only); broaden
   later.

3. Multi-output custom ops. The trait returns `Vec<ArgType>` for outputs.
   Handled naturally; no extra design needed.

4. `field()` and learnable params. The trait supports declaring a struct field.
   `collect_snapshots` is included from day one to avoid a breaking trait
   change later, and `create_lazy_snapshot` is exported so custom ops build
   snapshots the same way built-in nodes do.

5. Versioning of the `CustomOp` trait. It lives in `burn-onnx`'s public API.
   Any change is a semver event. The trait above is minimal on purpose;
   defaulted methods leave room to add capability later without breaking
   existing impls. The same applies to the promoted `onnx-ir` re-exports
   (`NodeType`, `ProcessError`, `CustomNode`): they become semver surface for
   `onnx-ir`, which is the cost of letting users implement inference.

6. `CodegenContext` / `Imports` wrappers vs exposing modules directly. The
   wrappers hide `ScopeAtPosition::scope()` (which returns `&mut Scope` and
   would transitively freeze the clone-tracking/partitioning API) and
   `BurnImports`' internals. The cheaper alternative is to make `mod scope` /
   `mod imports` / `mod node_traits` public and re-export the types. That is
   less code now but freezes a much larger surface as semver-stable. Recommend
   the wrappers unless the maintainers prefer to defer the API-stability cost.

7. Standard-domain list. `is_standard_domain` hard-codes `""`, `"ai.onnx"`,
   `"ai.onnx.ml"`. ONNX also defines `ai.onnx.training` and
   `ai.onnx.preview.training`, and `com.microsoft` is common in
   onnxruntime-optimized models. Treating all of those as Custom is correct
   today (this crate implements no built-in ops from them) and gives
   `com.microsoft` contrib ops a hook path for free; a future built-in training
   op would need a list update.

8. Should custom ops be able to declare `input_preferences` (e.g. request a
   `ScalarNative` input the way `Range` / `Slice` do)? V1: no; inputs arrive as
   whatever the producer inferred, and `OutputPreferences` on custom outputs are
   likewise ignored (4.4). A defaulted trait method can add either later without
   breaking impls.

9. Static-method vs instance-based traits: resolved in revision 2 in favor of
   `&self` (object safety kills the erased-bridging layer; hooks can carry
   configuration). Recorded here because v1 chose the opposite.

10. Testability of `forward`. `CodegenContext` is deliberately unforgeable
    (`pub(crate)` constructor), which is right for semver but means a hook
    author cannot call their own `forward` in a unit test; the only iteration
    loop is a full `ModelGen` run plus reading the generated file.
    `CustomNode::new` solved this for `infer_output_types`. A
    `CodegenContext` test-support constructor (backed by a throwaway scope,
    possibly behind a `test-support` feature) would close the gap without
    widening the semver surface much.

11. A declarative layer for the common case. The 90% hook is "call function F
    with the inputs in order, plus attributes X and Y" - yet today that costs
    learning `quote!`, the `ctx.arg` vs `arg_to_ident` split, and the
    output-binding convention. A convenience registration that maps
    `(op_type, domain)` to a function path plus an argument spec (inputs in
    order, named attributes, device) could be built entirely on top of
    `CustomOp` later, collapsing the common case to a few declarative lines
    with no proc-macro exposure. The full trait remains the escape hatch.

12. Build-time-only constant inputs stay live. A hook that reads a constant
    input via `Argument::value()` and inlines the values (the `ChannelScale`
    op in `examples/custom-op-hooks/`) leaves nothing consuming that input at
    runtime - but the graph still sees the custom node as a consumer, so the
    initializer is lifted to a `Param` field, written into the `.bpk`, and
    bound by an unused `let ... = self.constantN.val();` line that trips
    `unused_variables` in the user's build. Harmless, but it wastes weight-file
    space and puts a warning in code the user cannot edit. A fix needs the hook
    to declare which inputs it consumes at build time (e.g. a
    `build_time_inputs()` method) so the pipeline can drop them; that is an API
    addition, deferred rather than bolted on. (Unused bindings from
    multi-output built-ins - dropout masks, LSTM states - are a pre-existing,
    separate instance of the same warning class.)

13. Config parsed once. `infer_output_types` and `forward` both parse the same
    attributes (validation at parse time, values at codegen time); built-in
    nodes avoid this via `extract_config`, custom ops cannot because the two
    calls happen in different phases on different node snapshots. The
    documented v1 pattern is a shared `fn parse_config(&CustomNode) ->
    Result<Config, ProcessError>` called from both. A cached-config mechanism
    (e.g. the trait declaring an associated config type produced once at
    parse and replayed at codegen) would remove the duplication but couples
    the phases; deferred until real hooks show the pain is worth it.

## 10. References

- Issue: <https://github.com/tracel-ai/burn-onnx/issues/23>
- PyTorch symbolic registration tutorial:
  <https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html>
- Related onnxruntime upstream issue:
  <https://github.com/microsoft/onnxruntime/issues/27796>
