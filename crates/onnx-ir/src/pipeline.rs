//! ONNX to IR conversion pipeline orchestrator
//!
//! This module provides the high-level orchestration of the ONNX conversion process.
//! It clearly shows the entire conversion flow from start to finish.
//!
//! # Zero-Copy Loading
//!
//! When the `mmap` feature is enabled (default), files are memory-mapped for zero-copy
//! tensor loading. This significantly reduces memory usage for large models.
//!
//! # Usage
//!
//! ```ignore
//! use onnx_ir::OnnxGraphBuilder;
//!
//! // Build from file
//! let graph = OnnxGraphBuilder::new().parse_file("model.onnx")?;
//!
//! // Build from bytes
//! let graph = OnnxGraphBuilder::new().parse_bytes(&bytes)?;
//!
//! // Build from reader
//! let graph = OnnxGraphBuilder::new().parse_reader(file)?;
//! ```

use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use std::{fmt, fs::File, path::Path};

use protobuf::Message;

use crate::{
    ir::{NodeType, OnnxGraph},
    node::custom::{CustomOpInference, HookedCustomProcessor},
    processor::ProcessError,
    protos::ModelProto,
    registry::{ProcessorMethods, ProcessorRegistry},
};

use super::phases::{
    finalization, initialization, node_conversion, post_processing, type_inference,
};

/// Per-parse hook state threaded through the pipeline phases.
///
/// The global processor registry cannot carry user hooks (it is a shared
/// singleton), so the type-inference phase resolves processors through this
/// overlay instead: `NodeType::Custom` gets the hook-aware processor, every
/// other node type falls through to the global registry.
pub(crate) struct PipelineHooks {
    /// Owns the user hook (`None` = no hooks); the sole storage of the Arc.
    custom: HookedCustomProcessor,
}

impl PipelineHooks {
    pub(crate) fn new(inference: Option<Arc<dyn CustomOpInference>>) -> Self {
        Self {
            custom: HookedCustomProcessor::new(inference),
        }
    }

    /// The registered inference hook, for capture into `DeferredGraph` (so
    /// subgraph builds re-enter the pipeline with it).
    pub(crate) fn inference(&self) -> Option<Arc<dyn CustomOpInference>> {
        self.custom.hooks()
    }

    /// Resolution point used by the type-inference phase in place of
    /// `registry.get(node_type)`.
    pub(crate) fn resolve<'a>(
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

/// A custom op present in the model that no registered hook covers.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MissingHook {
    /// Raw ONNX op_type of the uncovered operator.
    pub op_type: String,
    /// Raw ONNX domain ("" = default domain).
    pub domain: String,
    /// Number of nodes in the graph using this operator.
    pub node_count: usize,
    /// The opset the model imports for the operator's domain.
    pub model_opset: usize,
    /// Why the operator is uncovered.
    pub reason: crate::node::custom::MissingReason,
}

impl MissingHook {
    /// Construct a missing-hook diagnostic (the pipeline's coverage pre-pass
    /// is the normal producer; this is public for tests and tooling).
    pub fn new(
        op_type: impl Into<String>,
        domain: impl Into<String>,
        node_count: usize,
        model_opset: usize,
        reason: crate::node::custom::MissingReason,
    ) -> Self {
        Self {
            op_type: op_type.into(),
            domain: domain.into(),
            node_count,
            model_opset,
            reason,
        }
    }
}

impl fmt::Display for MissingHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.domain.is_empty() {
            write!(f, "{}", self.op_type)?;
        } else {
            write!(f, "{}::{}", self.domain, self.op_type)?;
        }
        write!(f, " used by {} node(s)", self.node_count)?;
        if let crate::node::custom::MissingReason::OpsetMismatch { supported } = &self.reason {
            write!(
                f,
                " (hook covers opsets {supported}, model uses {})",
                self.model_opset
            )?;
        }
        Ok(())
    }
}

/// Errors that can occur when parsing ONNX models
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Failed to open or read the ONNX file
    Io { path: String, error: std::io::Error },

    /// Failed to parse ONNX protobuf format
    InvalidFormat { path: Option<String>, error: String },

    /// Model graph nodes are not topologically sorted (ONNX spec violation)
    InvalidGraphStructure { reason: String },

    /// Model uses default-domain operators without importing the default domain
    MissingOpsetVersion,

    /// Type inference failed during IR conversion
    TypeInference(ProcessError),

    /// Generic processing error
    Processing(ProcessError),

    /// Custom ops present that no registered inference hook covers.
    ///
    /// Only raised when hooks are registered (via
    /// [`OnnxGraphBuilder::with_custom_op_inference`]); a hook-less parse
    /// keeps the tolerant same-as-input fallback for inspection.
    MissingCustomOpHooks(Vec<MissingHook>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, error } => {
                write!(f, "Failed to open ONNX file '{}': {}", path, error)
            }
            Error::InvalidFormat { path, error } => {
                if let Some(p) = path {
                    write!(f, "Invalid ONNX format in '{}': {}", p, error)
                } else {
                    write!(f, "Invalid ONNX format: {}", error)
                }
            }
            Error::InvalidGraphStructure { reason } => {
                write!(f, "Invalid ONNX graph structure: {}", reason)
            }
            Error::MissingOpsetVersion => {
                write!(
                    f,
                    "ONNX model uses default-domain operators but its opset_import \
                     declares no version for the default domain (\"\" or \"ai.onnx\")"
                )
            }
            Error::TypeInference(e) => {
                write!(f, "Type inference failed: {e}")
            }
            Error::Processing(e) => {
                write!(f, "Processing error: {e}")
            }
            Error::MissingCustomOpHooks(missing) => {
                // No trailing "register a hook" instruction: the actionable
                // hint names an API this layer does not know about, so the
                // caller appends it (ModelGen points at register_custom_op).
                write!(
                    f,
                    "model contains {} custom op(s) with no covering inference hook:",
                    missing.len()
                )?;
                for hook in missing {
                    write!(f, "\n  - {hook}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl From<ProcessError> for Error {
    fn from(error: ProcessError) -> Self {
        Error::Processing(error)
    }
}

/// ONNX IR builder with fluent API
///
/// Builds ONNX IR graphs from various sources (files, bytes, readers).
/// Future configuration options can be added without breaking changes.
///
/// # Examples
///
/// ```ignore
/// use onnx_ir::OnnxGraphBuilder;
///
/// // Build from file (uses mmap when feature is enabled)
/// let graph = OnnxGraphBuilder::new().parse_file("model.onnx")?;
///
/// // Build from bytes
/// let graph = OnnxGraphBuilder::new().parse_bytes(&model_bytes)?;
///
/// // Build from reader
/// let graph = OnnxGraphBuilder::new().parse_reader(std::io::Cursor::new(data))?;
/// ```
#[derive(Clone)]
pub struct OnnxGraphBuilder {
    /// Whether to run graph simplification passes (default: true)
    simplify: bool,
    /// Type-inference hooks for custom (non-built-in) operators
    custom_op_inference: Option<Arc<dyn CustomOpInference>>,
}

impl fmt::Debug for OnnxGraphBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxGraphBuilder")
            .field("simplify", &self.simplify)
            .field(
                "custom_op_inference",
                &self.custom_op_inference.as_ref().map(|_| "<hooks>"),
            )
            .finish()
    }
}

impl Default for OnnxGraphBuilder {
    fn default() -> Self {
        Self {
            simplify: true,
            custom_op_inference: None,
        }
    }
}

impl OnnxGraphBuilder {
    /// Create a new ONNX graph builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable graph simplification passes (default: true)
    ///
    /// When enabled, the builder runs optimization passes on the IR graph
    /// such as dead node elimination, common subexpression elimination, and
    /// pattern-based simplifications.
    pub fn simplify(mut self, simplify: bool) -> Self {
        self.simplify = simplify;
        self
    }

    /// Register type-inference hooks for custom (non-built-in) operators.
    ///
    /// During type inference, `NodeType::Custom` nodes are resolved through
    /// the given hooks instead of the best-effort same-as-input fallback.
    /// Subgraph builds (If/Loop/Scan bodies) inherit the hooks.
    pub fn with_custom_op_inference(mut self, hooks: Arc<dyn CustomOpInference>) -> Self {
        self.custom_op_inference = Some(hooks);
        self
    }

    /// Parse an ONNX model from a file path
    ///
    /// When the `mmap` feature is enabled (default), the file is memory-mapped
    /// for zero-copy tensor loading, significantly reducing memory usage.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened or read
    /// - File is not valid ONNX protobuf format
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_file(self, path: impl AsRef<Path>) -> Result<OnnxGraph, Error> {
        let path = path.as_ref();
        log::info!("Parsing ONNX file: {}", path.display());

        // Load file contents - mmap when feature is enabled
        #[cfg(feature = "mmap")]
        let buffer = {
            let file = File::open(path).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            // SAFETY: We're mapping a read-only file. The bytes::Bytes keeps
            // the mmap alive for as long as tensor data references it.
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            log::debug!("Memory-mapped ONNX file ({} bytes)", mmap.len());
            bytes::Bytes::from_owner(mmap)
        };

        #[cfg(not(feature = "mmap"))]
        let buffer = {
            let mut file = File::open(path).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            log::debug!("Read ONNX file into memory ({} bytes)", buf.len());
            bytes::Bytes::from(buf)
        };

        self.parse_buffer(buffer, Some(path))
    }

    /// Parse an ONNX model from a byte slice
    ///
    /// Note: This copies the data internally. For large models already in memory
    /// as `bytes::Bytes`, consider using the internal buffer directly.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data is not valid ONNX protobuf format
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_bytes(self, data: &[u8]) -> Result<OnnxGraph, Error> {
        let buffer = bytes::Bytes::copy_from_slice(data);
        self.parse_buffer(buffer, None)
    }

    /// Parse an ONNX model from a reader
    ///
    /// Reads all data into memory before parsing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Reading from the reader fails
    /// - Data is not valid ONNX protobuf format
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_reader<R: Read>(self, mut reader: R) -> Result<OnnxGraph, Error> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).map_err(|error| Error::Io {
            path: "<reader>".to_string(),
            error,
        })?;
        log::debug!("Read ONNX from reader ({} bytes)", buf.len());
        let buffer = bytes::Bytes::from(buf);
        self.parse_buffer(buffer, None)
    }

    /// Internal: Parse from a bytes::Bytes buffer
    fn parse_buffer(
        self,
        buffer: bytes::Bytes,
        source_path: Option<&Path>,
    ) -> Result<OnnxGraph, Error> {
        let path_str = source_path.map(|p| p.display().to_string());

        // Get the base directory for external data resolution
        let base_path = source_path.and_then(|p| p.parent());

        let model: ModelProto =
            Message::parse_from_tokio_bytes(&buffer).map_err(|e| Error::InvalidFormat {
                path: path_str.clone(),
                error: e.to_string(),
            })?;

        // ONNX nodes must be topologically sorted per spec:
        // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
        if !model.graph.node.is_top_sorted() {
            return Err(Error::InvalidGraphStructure {
                reason: "Nodes are not topologically sorted (ONNX spec violation)".to_string(),
            });
        }

        let hooks = PipelineHooks::new(self.custom_op_inference.clone());
        let graph = build_graph_with_options(&model, base_path, self.simplify, &hooks)?;

        if let Some(path) = path_str {
            log::info!("Finished parsing ONNX file: {}", path);
        } else {
            log::info!("Finished parsing ONNX from bytes");
        }
        Ok(graph)
    }
}

/// Build IR graph from ONNX model with base path and simplification option
fn build_graph_with_options(
    model: &ModelProto,
    base_path: Option<&Path>,
    simplify: bool,
    hooks: &PipelineHooks,
) -> Result<OnnxGraph, Error> {
    let (opset_version, domain_opsets) = extract_opset_versions(model)?;
    let graph_builder = build_graph_builder_from_proto(
        &model.graph,
        opset_version,
        &domain_opsets,
        None,
        base_path,
        simplify,
        hooks,
    )?;

    log::debug!(" PHASE 6: Node Conversion (RawNode -> Node) ");
    Ok(graph_builder.convert_to_graph(opset_version))
}

/// Build IR graph as OnnxGraphBuilder from ONNX GraphProto
///
/// This returns OnnxGraphBuilder which still contains RawNode instances.
/// Call convert_to_graph() to get the final OnnxGraph with Node enum instances.
///
/// # Errors
///
/// Returns an error if node conversion or type inference fails
pub(crate) fn build_graph_builder_from_proto(
    graph: &crate::protos::GraphProto,
    opset_version: usize,
    domain_opsets: &DomainOpsets,
    name_registry: Option<crate::graph_state::NameRegistry>,
    base_path: Option<&Path>,
    simplify: bool,
    hooks: &PipelineHooks,
) -> Result<crate::ir::OnnxGraphBuilder, Error> {
    build_graph_builder_from_proto_with_outer_scope(
        graph,
        opset_version,
        domain_opsets,
        name_registry,
        crate::ir::OuterScopeTypes::new(),
        base_path,
        simplify,
        hooks,
    )
}

/// Build IR graph as OnnxGraphBuilder with access to outer scope types
///
/// This is used for building subgraphs that reference values from parent graphs.
/// The `outer_scope` map provides types for values that the subgraph references
/// but doesn't define internally.
///
/// The `base_path` is the directory containing the ONNX file, used for resolving
/// external tensor data paths (for models >2GB).
///
/// # Errors
///
/// Returns an error if node conversion or type inference fails
#[allow(clippy::too_many_arguments)] // internal per-parse context, threaded explicitly
pub(crate) fn build_graph_builder_from_proto_with_outer_scope(
    graph: &crate::protos::GraphProto,
    opset_version: usize,
    domain_opsets: &DomainOpsets,
    name_registry: Option<crate::graph_state::NameRegistry>,
    outer_scope: crate::ir::OuterScopeTypes,
    base_path: Option<&Path>,
    simplify: bool,
    hooks: &PipelineHooks,
) -> Result<crate::ir::OnnxGraphBuilder, Error> {
    log::debug!(" PHASE 1: Initialization ");
    let state_rc = initialization::initialize_from_graph_with_registry_and_outer_scope(
        graph,
        name_registry,
        outer_scope,
        base_path,
    );

    log::debug!(" PHASE 2: Node Conversion (Proto -> RawNode) ");
    node_conversion::convert_nodes_from_graph(
        graph,
        &state_rc,
        opset_version,
        domain_opsets,
        hooks,
    )?;

    // Fold constant expressions (Slice, Concat, Unsqueeze, etc.) before type inference.
    // Models exported from PyTorch often split initializer weights via Slice+Concat+Unsqueeze
    // chains before feeding them into RNN nodes. Without folding, these remain Dynamic and
    // block type inference for downstream nodes that need constant weight shapes.
    log::debug!(" PHASE 2b: Early Constant Folding ");
    {
        let mut state = state_rc.borrow_mut();
        let mut nodes = std::mem::take(&mut state.processed_nodes);
        drop(state);

        // Run in a fixed-point loop so that Slice -> Concat -> Unsqueeze chains cascade
        let max_iterations = 10;
        let mut converged = false;
        for _ in 0..max_iterations {
            let const_count_before = nodes
                .iter()
                .filter(|n| n.node_type == crate::ir::NodeType::Constant)
                .count();
            nodes = crate::simplify::constant_fold::fold_weight_rearrangements(nodes, &state_rc);
            let const_count_after = nodes
                .iter()
                .filter(|n| n.node_type == crate::ir::NodeType::Constant)
                .count();
            if const_count_after == const_count_before {
                converged = true;
                break;
            }
        }
        if !converged {
            log::debug!(
                "Early constant folding: reached max iterations ({max_iterations}) without converging"
            );
        }

        let mut state = state_rc.borrow_mut();
        state.processed_nodes = nodes;
    }

    // Coverage runs before type inference: an uncovered custom op would get
    // the same-as-input fallback there, and a possibly-wrong guessed type
    // then fails a *downstream* node with an unrelated cascade error before
    // any friendly summary could be produced. Hook-less parses skip this and
    // keep the tolerant fallback (useful for inspection/debugging).
    if let Some(inference) = hooks.inference() {
        log::debug!(" PHASE 2c: Custom-op coverage check ");
        check_custom_op_coverage(&state_rc.borrow().processed_nodes, inference.as_ref())?;
    }

    log::debug!(" PHASE 3: Type Inference ");
    type_inference::infer_types(&state_rc, opset_version, hooks).map_err(Error::TypeInference)?;

    log::debug!(" PHASE 4: Post-processing ");
    let (nodes, inputs, outputs) = post_processing::post_process(&state_rc, simplify);

    let (mut nodes, inputs, mut outputs) = if simplify {
        log::debug!(" PHASE 4b: Simplification ");
        crate::simplify::simplify_graph(nodes, inputs, outputs, &state_rc)
    } else {
        (nodes, inputs, outputs)
    };

    log::debug!(" PHASE 5: Finalization ");
    Ok(finalization::finalize(
        &mut nodes,
        inputs,
        &mut outputs,
        state_rc,
    ))
}

/// Phase 2c: verify every custom op is covered by the registered hooks.
///
/// Aggregates all uncovered `(domain, op_type)` pairs with usage counts so
/// the user sees the complete list at once instead of failing one op at a
/// time.
fn check_custom_op_coverage(
    nodes: &[crate::ir::RawNode],
    hooks: &dyn CustomOpInference,
) -> Result<(), Error> {
    use std::collections::BTreeMap;

    let mut missing: BTreeMap<(String, String), MissingHook> = BTreeMap::new();

    for node in nodes.iter().filter(|n| n.node_type == NodeType::Custom) {
        // Same parser invariant custom_node_view enforces: a violated
        // internal invariant must be loud, not silently skipped (a skipped
        // node would dodge the coverage gate here and panic later anyway).
        let identity = node
            .custom_identity
            .as_ref()
            .expect("RawNode with NodeType::Custom must carry a CustomIdentity (parser invariant)");
        match hooks.coverage(&identity.op_type, &identity.domain, identity.domain_opset) {
            crate::node::custom::HookCoverage::Covered => {}
            crate::node::custom::HookCoverage::Missing(reason) => {
                missing
                    .entry((identity.domain.clone(), identity.op_type.clone()))
                    .and_modify(|m| m.node_count += 1)
                    .or_insert_with(|| MissingHook {
                        op_type: identity.op_type.clone(),
                        domain: identity.domain.clone(),
                        node_count: 1,
                        model_opset: identity.domain_opset,
                        reason,
                    });
            }
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(Error::MissingCustomOpHooks(missing.into_values().collect()))
    }
}

/// The `ai.onnx` spelling of the default ONNX operator set domain.
///
/// ONNX's own `onnx/common/constants.h` documents `""` and `"ai.onnx"` as
/// "equivalent in an onnx proto representation" and folds the latter into the
/// former via `NormalizeDomain`. Note that `ai.onnx.ml` is *not* an alias: it
/// is the separate ONNX-ML operator set with its own opset numbering.
///
/// `onnx.checker` does not apply this normalization (it rejects `ai.onnx` node
/// domains outright); ONNX Runtime does, and so do we.
const AI_ONNX_DOMAIN: &str = "ai.onnx";

/// Newest default-domain opset this crate implements, used only as the stand-in
/// for models that import no default domain at all.
///
/// Tracks the highest `min_opset` across the node processors (`Swish`, opset
/// 24). It does not have to be exact: it only has to sit above the version
/// gates in `extract_config`, so an opset-dependent operator gets current
/// semantics rather than the opset-1 legacy branch.
const LATEST_DEFAULT_OPSET: usize = 24;

/// Fold `ai.onnx` onto the canonical empty-string spelling of the default domain.
///
/// Public because operator identity is `(op_type, domain)` on both sides of a
/// custom-op hook: burn-onnx canonicalizes a hook's declared domain with this
/// so a hook registered for `""` covers a node whose model spells it `ai.onnx`.
pub fn normalize_domain(domain: &str) -> &str {
    if domain == AI_ONNX_DOMAIN { "" } else { domain }
}

/// Opset version for every domain listed in the model's `opset_import`.
///
/// ONNX operator identity is `(domain, op_type, opset-for-that-domain)`, so
/// custom-domain nodes must be tagged with their own domain's opset, not the
/// default ONNX opset. Wrapped in `Arc` for cheap cloning into `DeferredGraph`
/// (subgraphs inherit the model-level imports).
///
/// `new` guarantees a `""` entry and `opset_for` normalizes lookups, so `""`
/// and `ai.onnx` always resolve to the same version. That entry is not always a
/// real import: see `extract_opset_versions` for the models that have none.
#[derive(Debug, Clone)]
pub(crate) struct DomainOpsets {
    versions: Arc<HashMap<String, usize>>,
    default_opset: usize,
}

impl DomainOpsets {
    pub(crate) fn new(mut versions: HashMap<String, usize>, default_opset: usize) -> Self {
        // Enforce the documented invariant even for hand-built instances
        // (tests): the default domain is always present.
        versions.entry(String::new()).or_insert(default_opset);
        Self {
            versions: Arc::new(versions),
            default_opset,
        }
    }

    /// Opset version for `domain`, falling back to the default-domain opset.
    ///
    /// Per the ONNX spec every domain a node uses must appear in
    /// `opset_import`; the fallback is robustness against malformed exporters.
    pub(crate) fn opset_for(&self, domain: &str) -> usize {
        let domain = normalize_domain(domain);
        if let Some(version) = self.versions.get(domain) {
            return *version;
        }
        log::warn!(
            "Domain '{domain}' has no opset_import entry (malformed model); \
             falling back to default-domain opset {}",
            self.default_opset
        );
        self.default_opset
    }
}

/// Returns true if any node in `graph`, or in a nested subgraph, belongs to the
/// default domain.
///
/// Subgraphs normally hang off default-domain control flow (If/Loop/Scan), in
/// which case the parent node already answers this before the recursion runs.
/// The recursion is here for the custom-domain op that carries one anyway.
/// `AttributeProto` reaches a `GraphProto` only through `g` and `graphs`, so
/// those two cover every subgraph reachable from `graph`.
fn uses_default_domain(graph: &crate::protos::GraphProto) -> bool {
    graph.node.iter().any(|node| {
        normalize_domain(&node.domain).is_empty()
            || node.attribute.iter().any(|attr| {
                attr.g.as_ref().is_some_and(uses_default_domain)
                    || attr.graphs.iter().any(uses_default_domain)
            })
    })
}

/// Per-domain opset versions from `opset_import`, with `ai.onnx` folded onto
/// the default domain.
///
/// Both spellings map to the same key, so a model listing both is merged rather
/// than resolved by whichever entry protobuf happens to put last.
fn collect_domain_versions(model: &ModelProto) -> HashMap<String, usize> {
    let mut versions: HashMap<String, usize> = HashMap::new();
    for opset in &model.opset_import {
        let domain = normalize_domain(&opset.domain);
        let version = opset.version as usize;
        match versions.get(domain) {
            Some(&existing) if existing != version => {
                let kept = existing.max(version);
                // The default domain reaches here from two spellings as well as
                // from a plain duplicate, so name the alias only in that case.
                if domain.is_empty() {
                    log::warn!(
                        "The default domain is imported at conflicting opset \
                         versions {existing} and {version} (\"\" and \"ai.onnx\" \
                         name the same domain); using {kept}"
                    );
                } else {
                    log::warn!(
                        "Domain '{domain}' is imported at conflicting opset \
                         versions {existing} and {version}; using {kept}"
                    );
                }
                versions.insert(domain.to_string(), kept);
            }
            Some(_) => {}
            None => {
                versions.insert(domain.to_string(), version);
            }
        }
    }
    versions
}

/// Extract the model-level opset used to interpret every node, plus the
/// per-domain map that tags custom-domain nodes with their own domain's opset.
fn extract_opset_versions(model: &ModelProto) -> Result<(usize, DomainOpsets), Error> {
    let versions = collect_domain_versions(model);

    let default_opset = match versions.get("") {
        Some(version) => *version,
        // Importing the default domain is only required when the model actually
        // uses default-domain operators, so an ai.onnx.ml-only model (the H2O
        // export in issue #434) legitimately omits it.
        //
        // Stand in with the newest opset rather than anything derived from the
        // imports. Node specs are validated against this one model-level opset
        // regardless of domain, and opset numbering is per-domain: ai.onnx.ml
        // tops out around 5 and contrib domains sit at 1, so borrowing a number
        // from them would put every version gate in `extract_config` on its
        // pre-13 legacy branch (Softmax axis default, Squeeze axes-as-attribute,
        // Pad, Resize). Guessing "current" is the safer direction to be wrong in.
        None if !uses_default_domain(&model.graph) => {
            log::warn!(
                "Model imports no default-domain opset; assuming opset \
                 {LATEST_DEFAULT_OPSET} for node specs. Operators whose semantics \
                 depend on the default opset may be interpreted incorrectly."
            );
            LATEST_DEFAULT_OPSET
        }
        None => return Err(Error::MissingOpsetVersion),
    };

    Ok((default_opset, DomainOpsets::new(versions, default_opset)))
}

/// Trait for checking if a list of nodes is topologically sorted
pub(crate) trait TopologicalSortable {
    fn is_top_sorted(&self) -> bool;
}

use crate::protos::NodeProto;

impl TopologicalSortable for Vec<NodeProto> {
    fn is_top_sorted(&self) -> bool {
        // Iterate over each node in the vector
        for (node_position, node) in self.iter().enumerate() {
            // Iterate over each output of the node
            for output in &node.output {
                // If the output is empty, we don't want to check the rest of the graph, inputs and outputs that are optional
                // can end up as empty strings, so we can't use that as a reason to count the graph as not sorted
                if output.is_empty() {
                    continue;
                }
                // Iterate over each other node in the vector
                for (other_node_position, other_node) in self.iter().enumerate() {
                    // If the other node has an input that matches the current output
                    if other_node.input.contains(output) {
                        // If the position of the current node is greater than the position of the other node
                        if node_position > other_node_position {
                            // The vector is not topologically sorted
                            return false;
                        }
                    }
                }
            }
        }

        // The vector is topologically sorted
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, DType, Node};
    use crate::protos::{
        GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TypeProto, ValueInfoProto,
    };
    use crate::protos::{TensorShapeProto, tensor_shape_proto, type_proto};

    fn tensor_value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        let mut shape = TensorShapeProto::new();
        for d in dims {
            let mut dim = tensor_shape_proto::Dimension::new();
            dim.set_dim_value(*d);
            shape.dim.push(dim);
        }
        let mut tensor = type_proto::Tensor::new();
        tensor.elem_type = 1; // FLOAT
        tensor.shape = ::protobuf::MessageField::some(shape);
        let mut ty = TypeProto::new();
        ty.set_tensor_type(tensor);
        let mut vi = ValueInfoProto::new();
        vi.name = name.to_string();
        vi.type_ = ::protobuf::MessageField::some(ty);
        vi
    }

    /// A single-node model: input -> <domain>::<op_type> -> output
    fn single_node_model(domain: &str, op_type: &str, opset_imports: &[(&str, i64)]) -> Vec<u8> {
        let mut node = NodeProto::new();
        node.name = "the_node".to_string();
        node.op_type = op_type.to_string();
        node.domain = domain.to_string();
        node.input.push("input".to_string());
        node.output.push("output".to_string());

        let mut graph = GraphProto::new();
        graph.name = "test_graph".to_string();
        graph.input.push(tensor_value_info("input", &[2, 3]));
        graph.output.push(tensor_value_info("output", &[2, 3]));
        graph.node.push(node);

        let mut model = ModelProto::new();
        model.graph = ::protobuf::MessageField::some(graph);
        for (domain, version) in opset_imports {
            let mut op = OperatorSetIdProto::new();
            op.domain = domain.to_string();
            op.version = *version;
            model.opset_import.push(op);
        }
        model.write_to_bytes().unwrap()
    }

    fn parse_single_custom(bytes: &[u8]) -> crate::node::custom::CustomNode {
        let graph = OnnxGraphBuilder::new().parse_bytes(bytes).unwrap();
        graph
            .nodes
            .iter()
            .find_map(|n| match n {
                Node::Custom(c) => Some(c.clone()),
                _ => None,
            })
            .expect("graph should contain a Custom node")
    }

    #[test]
    fn custom_domain_node_parses_as_custom() {
        let bytes = single_node_model(
            "custom.domain",
            "FftLike",
            &[("", 16), ("custom.domain", 2)],
        );
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.op_type, "FftLike");
        assert_eq!(custom.domain, "custom.domain");
        // Domain-specific opset, not the default ONNX opset
        assert_eq!(custom.opset, 2);
        // Fallback inference: output type mirrors input
        assert!(matches!(
            &custom.outputs[0].ty,
            ArgType::Tensor(t) if t.dtype == DType::F32 && t.rank == 2
        ));
    }

    #[test]
    fn unknown_default_domain_op_parses_as_custom() {
        let bytes = single_node_model("", "TotallyUnknownOp", &[("", 16)]);
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.op_type, "TotallyUnknownOp");
        assert_eq!(custom.domain, "");
        assert_eq!(custom.opset, 16);
    }

    #[test]
    fn builtin_name_in_custom_domain_stays_custom() {
        // ONNX identity is (domain, op_type): my.domain::MatMul is NOT the
        // built-in MatMul and must not silently get default-domain semantics.
        let bytes = single_node_model("my.domain", "MatMul", &[("", 16), ("my.domain", 5)]);
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.op_type, "MatMul");
        assert_eq!(custom.domain, "my.domain");
        assert_eq!(custom.opset, 5);
    }

    #[test]
    fn missing_domain_import_falls_back_to_default_opset() {
        // Malformed model: the node's domain has no opset_import entry.
        let bytes = single_node_model("no.import.domain", "SomeOp", &[("", 16)]);
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.opset, 16);
    }

    #[test]
    fn builtin_op_still_resolves_normally() {
        let bytes = single_node_model("", "Relu", &[("", 16)]);
        let graph = OnnxGraphBuilder::new().parse_bytes(&bytes).unwrap();
        assert!(graph.nodes.iter().any(|n| matches!(n, Node::Relu(_))));
        assert!(!graph.nodes.iter().any(|n| matches!(n, Node::Custom(_))));
    }

    #[test]
    fn ai_onnx_import_is_the_default_domain() {
        // "" and "ai.onnx" name the same operator set, so an "ai.onnx" entry
        // satisfies a default-domain node's need for an opset.
        let bytes = single_node_model("", "Relu", &[("ai.onnx", 16)]);
        let graph = OnnxGraphBuilder::new().parse_bytes(&bytes).unwrap();
        assert!(graph.nodes.iter().any(|n| matches!(n, Node::Relu(_))));

        // Relu's spec is opset-permissive, so also pin that the version itself
        // came out of the "ai.onnx" entry rather than a fallback.
        let bytes = single_node_model("no.import.domain", "SomeOp", &[("ai.onnx", 16)]);
        assert_eq!(parse_single_custom(&bytes).opset, 16);
    }

    #[test]
    fn duplicate_default_domain_spellings_merge_by_max() {
        // onnx.checker accepts a model importing both spellings at different
        // versions. Folding them into one key must not leave the answer up to
        // whichever entry protobuf happens to put last.
        for imports in [[("", 13), ("ai.onnx", 17)], [("ai.onnx", 17), ("", 13)]] {
            let bytes = single_node_model("no.import.domain", "SomeOp", &imports);
            assert_eq!(
                parse_single_custom(&bytes).opset,
                17,
                "import order {imports:?} changed the resolved opset"
            );
        }
    }

    #[test]
    fn duplicate_non_default_domain_imports_merge_by_max() {
        // Not an alias collision, just a model importing one domain twice.
        // onnx.checker accepts it, and it takes the same merge path.
        for imports in [
            [("", 16), ("custom.domain", 2), ("custom.domain", 5)],
            [("", 16), ("custom.domain", 5), ("custom.domain", 2)],
        ] {
            let bytes = single_node_model("custom.domain", "SomeOp", &imports);
            assert_eq!(
                parse_single_custom(&bytes).opset,
                5,
                "import order {imports:?} changed the resolved opset"
            );
        }
    }

    #[test]
    fn no_imports_at_all_uses_the_latest_opset() {
        // Degenerate but reachable: nothing to derive a version from, and the
        // stand-in must not land on the opset-1 legacy branch.
        let bytes = single_node_model("custom.domain", "SomeOp", &[]);
        assert_eq!(parse_single_custom(&bytes).opset, LATEST_DEFAULT_OPSET);
    }

    #[test]
    fn ai_onnx_node_domain_gets_the_default_opset() {
        let bytes = single_node_model("ai.onnx", "TotallyUnknownOp", &[("", 16)]);
        let custom = parse_single_custom(&bytes);

        // No spurious "domain has no opset_import entry" fallback: "ai.onnx"
        // resolves through the default domain's entry.
        assert_eq!(custom.opset, 16);
        // Identity is canonicalized, so a hook registered for the default
        // domain covers this node.
        assert_eq!(custom.domain, "");
    }

    #[test]
    fn ml_only_model_needs_no_default_domain_import() {
        // The OnnxMLTools H2O export in issue #434 imports only ai.onnx.ml,
        // its graph using no default-domain operators. onnx.checker accepts
        // that, so parsing must not demand a default-domain opset.
        let bytes = single_node_model("ai.onnx.ml", "TreeEnsembleRegressor", &[("ai.onnx.ml", 1)]);
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.op_type, "TreeEnsembleRegressor");
        assert_eq!(custom.domain, "ai.onnx.ml");
        assert_eq!(custom.opset, 1);
    }

    #[test]
    fn default_domain_node_without_import_errors() {
        // Genuinely malformed: guessing an opset here would silently pick the
        // wrong schema version for the operator.
        let bytes = single_node_model("", "Relu", &[("ai.onnx.ml", 1)]);
        let err = OnnxGraphBuilder::new().parse_bytes(&bytes).unwrap_err();

        assert!(matches!(err, Error::MissingOpsetVersion), "got: {err}");
    }

    /// A custom-domain node carrying a subgraph whose body holds a
    /// default-domain Relu, attached via either the `g` or the `graphs` field.
    fn subgraph_model(use_graphs_field: bool, opset_imports: &[(&str, i64)]) -> Vec<u8> {
        let mut inner_node = NodeProto::new();
        inner_node.name = "inner".to_string();
        inner_node.op_type = "Relu".to_string();
        inner_node.input.push("input".to_string());
        inner_node.output.push("inner_output".to_string());

        let mut inner_graph = GraphProto::new();
        inner_graph.name = "inner_graph".to_string();
        inner_graph.node.push(inner_node);

        let mut attr = crate::protos::AttributeProto::new();
        attr.name = "body".to_string();
        if use_graphs_field {
            attr.type_ = ::protobuf::EnumOrUnknown::new(
                crate::protos::attribute_proto::AttributeType::GRAPHS,
            );
            attr.graphs.push(inner_graph);
        } else {
            attr.type_ = ::protobuf::EnumOrUnknown::new(
                crate::protos::attribute_proto::AttributeType::GRAPH,
            );
            attr.g = ::protobuf::MessageField::some(inner_graph);
        }

        let mut node = NodeProto::new();
        node.name = "outer".to_string();
        node.op_type = "SomeOp".to_string();
        node.domain = "custom.domain".to_string();
        node.input.push("input".to_string());
        node.output.push("output".to_string());
        node.attribute.push(attr);

        let mut graph = GraphProto::new();
        graph.name = "test_graph".to_string();
        graph.input.push(tensor_value_info("input", &[2, 3]));
        graph.output.push(tensor_value_info("output", &[2, 3]));
        graph.node.push(node);

        let mut model = ModelProto::new();
        model.graph = ::protobuf::MessageField::some(graph);
        for (domain, version) in opset_imports {
            let mut op = OperatorSetIdProto::new();
            op.domain = domain.to_string();
            op.version = *version;
            model.opset_import.push(op);
        }
        model.write_to_bytes().unwrap()
    }

    #[test]
    fn default_domain_node_nested_in_subgraph_errors() {
        // The outer node's domain alone would clear the guard. Without the
        // recursion the model parses and the nested Relu gets interpreted at
        // a guessed opset instead of being rejected.
        for use_graphs_field in [false, true] {
            let bytes = subgraph_model(use_graphs_field, &[("custom.domain", 3)]);
            let err = OnnxGraphBuilder::new().parse_bytes(&bytes).unwrap_err();

            assert!(
                matches!(err, Error::MissingOpsetVersion),
                "graphs field: {use_graphs_field}, got: {err}"
            );
        }
    }

    #[test]
    fn custom_domain_subgraph_without_default_domain_nodes_parses() {
        // Same shape, but the subgraph holds no default-domain node, so the
        // guard must not fire. Pins that the recursion checks node domains
        // rather than merely the presence of a subgraph.
        for use_graphs_field in [false, true] {
            let bytes = subgraph_model(use_graphs_field, &[("custom.domain", 3), ("", 16)]);
            let graph = OnnxGraphBuilder::new().parse_bytes(&bytes).unwrap();

            assert!(graph.nodes.iter().any(|n| matches!(n, Node::Custom(_))));
        }
    }

    /// Hook that gives every custom op a fixed F64 rank-3 output type.
    struct RankThreeInference;

    impl crate::node::custom::CustomOpInference for RankThreeInference {
        fn coverage(
            &self,
            _op_type: &str,
            _domain: &str,
            _opset: usize,
        ) -> crate::node::custom::HookCoverage {
            crate::node::custom::HookCoverage::Covered
        }

        fn infer(
            &self,
            node: &crate::node::custom::CustomNode,
        ) -> Result<Option<Vec<ArgType>>, ProcessError> {
            Ok(Some(vec![
                ArgType::Tensor(crate::ir::TensorType::new(
                    DType::F64,
                    3,
                    None
                ));
                node.outputs.len()
            ]))
        }
    }

    #[test]
    fn registered_inference_hook_overrides_fallback() {
        let bytes = single_node_model(
            "custom.domain",
            "FftLike",
            &[("", 16), ("custom.domain", 2)],
        );
        // Simplify off: the graph output type check would reject the
        // rank change against the declared rank-2 graph output otherwise.
        let graph = OnnxGraphBuilder::new()
            .simplify(false)
            .with_custom_op_inference(Arc::new(RankThreeInference))
            .parse_bytes(&bytes)
            .unwrap();

        let custom = graph
            .nodes
            .iter()
            .find_map(|n| match n {
                Node::Custom(c) => Some(c),
                _ => None,
            })
            .expect("graph should contain a Custom node");

        // Hook-provided type, not the same-as-input (F32 rank-2) fallback
        assert!(matches!(
            &custom.outputs[0].ty,
            ArgType::Tensor(t) if t.dtype == DType::F64 && t.rank == 3
        ));
    }

    /// Inference with no coverage for anything (empty registry equivalent).
    struct NoCoverageInference;

    impl crate::node::custom::CustomOpInference for NoCoverageInference {
        fn coverage(
            &self,
            _op_type: &str,
            _domain: &str,
            _opset: usize,
        ) -> crate::node::custom::HookCoverage {
            crate::node::custom::HookCoverage::Missing(crate::node::custom::MissingReason::NoHook)
        }

        fn infer(
            &self,
            _node: &crate::node::custom::CustomNode,
        ) -> Result<Option<Vec<ArgType>>, ProcessError> {
            Ok(None)
        }
    }

    #[test]
    fn coverage_pre_pass_reports_uncovered_ops() {
        let bytes = single_node_model(
            "custom.domain",
            "FftLike",
            &[("", 16), ("custom.domain", 2)],
        );
        let err = OnnxGraphBuilder::new()
            .with_custom_op_inference(Arc::new(NoCoverageInference))
            .parse_bytes(&bytes)
            .unwrap_err();

        assert!(matches!(err, Error::MissingCustomOpHooks(_)));
        let msg = format!("{err}");
        assert!(msg.contains("custom.domain::FftLike"), "got: {msg}");
        assert!(msg.contains("used by 1 node(s)"), "got: {msg}");
        // The actionable "register a hook via ..." hint belongs to the caller
        // (ModelGen appends it); this layer only reports what is missing.
        assert!(!msg.contains("register"), "got: {msg}");
        assert!(msg.trim_end() == msg, "message must not end in whitespace");
    }

    #[test]
    fn coverage_pre_pass_reports_opset_mismatch() {
        struct MismatchInference;

        impl crate::node::custom::CustomOpInference for MismatchInference {
            fn coverage(
                &self,
                _op_type: &str,
                _domain: &str,
                _opset: usize,
            ) -> crate::node::custom::HookCoverage {
                crate::node::custom::HookCoverage::Missing(
                    crate::node::custom::MissingReason::OpsetMismatch {
                        supported: crate::node::custom::OpsetRange {
                            min: 1,
                            max: Some(1),
                        },
                    },
                )
            }

            fn infer(
                &self,
                _node: &crate::node::custom::CustomNode,
            ) -> Result<Option<Vec<ArgType>>, ProcessError> {
                Ok(None)
            }
        }

        let bytes = single_node_model(
            "custom.domain",
            "FftLike",
            &[("", 16), ("custom.domain", 3)],
        );
        let err = OnnxGraphBuilder::new()
            .with_custom_op_inference(Arc::new(MismatchInference))
            .parse_bytes(&bytes)
            .unwrap_err();

        let msg = format!("{err}");
        assert!(
            msg.contains("hook covers opsets 1..=1, model uses 3"),
            "got: {msg}"
        );
    }

    #[test]
    fn hookless_parse_skips_coverage_pre_pass() {
        // No hooks registered: custom ops keep the tolerant fallback so the
        // graph stays inspectable (PR 1 behavior).
        let bytes = single_node_model(
            "custom.domain",
            "FftLike",
            &[("", 16), ("custom.domain", 2)],
        );
        assert!(OnnxGraphBuilder::new().parse_bytes(&bytes).is_ok());
    }

    #[test]
    fn hook_error_fails_the_parse() {
        struct FailingInference;

        impl crate::node::custom::CustomOpInference for FailingInference {
            fn coverage(
                &self,
                _op_type: &str,
                _domain: &str,
                _opset: usize,
            ) -> crate::node::custom::HookCoverage {
                crate::node::custom::HookCoverage::Covered
            }

            fn infer(
                &self,
                _node: &crate::node::custom::CustomNode,
            ) -> Result<Option<Vec<ArgType>>, ProcessError> {
                Err(ProcessError::MissingAttribute("n_fft".to_string()))
            }
        }

        let bytes = single_node_model(
            "custom.domain",
            "FftLike",
            &[("", 16), ("custom.domain", 2)],
        );
        let err = OnnxGraphBuilder::new()
            .with_custom_op_inference(Arc::new(FailingInference))
            .parse_bytes(&bytes)
            .unwrap_err();

        let msg = format!("{err}");
        assert!(msg.contains("n_fft"), "got: {msg}");
    }
}
