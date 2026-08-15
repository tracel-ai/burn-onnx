//! ONNX attribute values
//!
//! This module contains the AttributeValue enum which represents various types
//! of attributes that can be attached to ONNX nodes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use burn_tensor::TensorData;

use crate::ir::{OnnxGraph, OnnxGraphBuilder};
use crate::protos::GraphProto;

/// Deferred subgraph that is built lazily during type inference.
///
/// ## Why Deferred?
///
/// ONNX control flow nodes (If, Loop, Scan) contain subgraphs that can reference
/// values from the parent graph's scope. For example:
///
/// ```text
/// x = Conv(input, weights)      // x has type Tensor[F32, rank=4]
/// y = If(condition) {
///     then_branch: Add(x, bias)  // References 'x' from parent scope
///     else_branch: Mul(x, scale)
/// }
/// ```
///
/// The subgraph's `Add(x, bias)` needs to know the type of `x` for type inference,
/// but `x`'s type is only determined after processing the parent's `Conv` node.
///
/// ## Solution: Lazy Building
///
/// 1. **Parse phase**: Store the raw `GraphProto` in `DeferredGraph`
/// 2. **Type inference phase**: When processing If/Loop/Scan, outer-scope types are known
/// 3. **Build phase**: Call `build_graph_with_outer_scope(types)` with resolved types
///
/// This lazy evaluation pattern ensures subgraphs have access to all type information
/// from the parent scope when they are finally built.
#[derive(Clone)]
pub struct DeferredGraph {
    /// The raw ONNX GraphProto (wrapped in Arc for cheap cloning)
    pub proto: Arc<GraphProto>,
    /// The opset version to use when building the subgraph
    pub opset_version: usize,
    /// Per-domain opset versions (inherited from the model's opset_import)
    pub(crate) domain_opsets: crate::pipeline::DomainOpsets,
    /// Name registry for unique node naming across subgraphs
    pub name_registry: Option<crate::graph_state::NameRegistry>,
    /// Base path for resolving external tensor data (inherited from parent graph)
    pub base_path: Option<PathBuf>,
    /// Custom-op inference hooks inherited from the parent parse. Subgraph
    /// builds run inside the parent's type-inference phase, so this Arc is
    /// the only channel that carries hooks into them.
    pub(crate) custom_op_inference: Option<Arc<dyn crate::node::custom::CustomOpInference>>,
}

impl std::fmt::Debug for DeferredGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeferredGraph")
            .field("proto", &self.proto)
            .field("opset_version", &self.opset_version)
            .field("domain_opsets", &self.domain_opsets)
            .field("name_registry", &self.name_registry)
            .field("base_path", &self.base_path)
            .field(
                "custom_op_inference",
                &self.custom_op_inference.as_ref().map(|_| "<hooks>"),
            )
            .finish()
    }
}

/// A map of outer-scope value names to their resolved arguments (including type and value)
///
/// Uses full `Argument` instead of just `ArgType` to preserve constant values
/// for subgraphs that reference parent graph constants (e.g., LSTM weights).
pub type OuterScopeTypes = std::collections::HashMap<String, crate::ir::Argument>;

impl DeferredGraph {
    /// Build the subgraph from the deferred GraphProto with access to outer scope types.
    ///
    /// This should be called during type inference when all outer-scope
    /// references have been resolved. The `outer_scope` map provides types
    /// for values that the subgraph references from the parent graph.
    pub fn build_with_outer_scope(
        &self,
        outer_scope: OuterScopeTypes,
    ) -> Result<OnnxGraphBuilder, crate::pipeline::Error> {
        let hooks = crate::pipeline::PipelineHooks::new(self.custom_op_inference.clone());
        crate::pipeline::build_graph_builder_from_proto_with_outer_scope(
            &self.proto,
            self.opset_version,
            &self.domain_opsets,
            self.name_registry.clone(),
            outer_scope,
            self.base_path.as_deref(),
            false,
            &hooks,
        )
    }

    /// Build and finalize the subgraph into an OnnxGraph with outer scope types.
    pub fn build_graph_with_outer_scope(
        &self,
        outer_scope: OuterScopeTypes,
    ) -> Result<OnnxGraph, crate::pipeline::Error> {
        let builder = self.build_with_outer_scope(outer_scope)?;
        Ok(builder.convert_to_graph(self.opset_version))
    }

    /// Build the subgraph from the deferred GraphProto without outer scope types.
    ///
    /// Useful for simple subgraphs that don't reference outer-scope values.
    #[allow(dead_code)]
    pub fn build(&self) -> Result<OnnxGraphBuilder, crate::pipeline::Error> {
        self.build_with_outer_scope(OuterScopeTypes::new())
    }

    /// Build and finalize the subgraph into an OnnxGraph without outer scope types.
    #[allow(dead_code)]
    pub fn build_graph(&self) -> Result<OnnxGraph, crate::pipeline::Error> {
        let builder = self.build()?;
        Ok(builder.convert_to_graph(self.opset_version))
    }
}

/// The type of an attribute.
#[derive(Debug, Clone)]
pub(crate) enum AttributeValue {
    Float32(f32),
    Float32s(Vec<f32>),
    Int64(i64),
    Int64s(Vec<i64>),
    String(String),
    #[allow(dead_code)]
    Strings(Vec<String>),
    Tensor(TensorData),
    #[allow(dead_code)]
    Tensors(Vec<TensorData>),
    /// Deferred graph attribute - raw GraphProto to be built during type inference
    DeferredGraph(DeferredGraph),
    /// Multiple deferred graphs (for ONNX GRAPHS attributes)
    #[allow(dead_code)]
    DeferredGraphs(Vec<DeferredGraph>),
    /// Final graph after conversion (used in final Node enum)
    /// Note: Constructed via DeferredGraph::build_graph_with_outer_scope(), not directly
    #[allow(dead_code)]
    Graph(OnnxGraph),
    /// Multiple final graphs (for ONNX GRAPHS attributes)
    #[allow(dead_code)]
    Graphs(Vec<OnnxGraph>),
}

pub type Attributes = HashMap<String, AttributeValue>;

/// Scalar/tensor attribute values exposed to custom-op hooks.
///
/// Deliberately a separate enum from the internal `AttributeValue`: it has no
/// graph payloads (subgraph custom hooks are out of scope for v1) and no
/// `Rc`-backed state, so types embedding it stay `Send + Sync`. Graph-valued
/// attributes are kept as payload-free markers so hooks can detect them.
#[derive(Debug, Clone)]
enum PublicAttributeValue {
    Float32(f32),
    Float32s(Vec<f32>),
    Int64(i64),
    Int64s(Vec<i64>),
    String(String),
    Strings(Vec<String>),
    Tensor(TensorData),
    Tensors(Vec<TensorData>),
    Graph,
    Graphs,
}

/// The ONNX kind of an attribute, as exposed to custom-op hooks.
///
/// Lets a hook distinguish "attribute absent" from "attribute present with a
/// different type" (every typed getter returns `None` for both), and detect
/// graph-valued attributes whose payload is not exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttrKind {
    /// FLOAT
    Float32,
    /// FLOATS
    Float32s,
    /// INT
    Int64,
    /// INTS
    Int64s,
    /// STRING
    String,
    /// STRINGS
    Strings,
    /// TENSOR
    Tensor,
    /// TENSORS
    Tensors,
    /// GRAPH: present on the node, but the subgraph payload is not exposed
    /// to hooks (subgraph custom hooks are out of scope).
    Graph,
    /// GRAPHS: like [`AttrKind::Graph`], for a list of subgraphs.
    Graphs,
}

/// Read-only, owned view of a node's ONNX attributes for custom ops.
///
/// Exposes typed getters so the internal `AttributeValue` enum (which carries
/// crate-private graph variants) stays private. Graph-valued attributes
/// (`GRAPH` / `GRAPHS`) appear in [`names`](Self::names) and
/// [`kind`](Self::kind) but their payload is not exposed.
///
/// Every typed getter returns `None` both when the attribute is absent and
/// when it is present with a different ONNX type; use
/// [`kind`](Self::kind) to distinguish (an exporter emitting e.g. `axis` as
/// FLOAT instead of INT should be a detectable error, not a silent default).
#[derive(Debug, Clone, Default)]
pub struct PublicAttributesOwned(HashMap<String, PublicAttributeValue>);

impl PublicAttributesOwned {
    /// Snapshot the internal attribute map. Graph-valued attributes keep
    /// only a payload-free marker.
    pub(crate) fn from_internal(attrs: &Attributes) -> Self {
        let map = attrs
            .iter()
            .map(|(name, value)| {
                let public = match value {
                    AttributeValue::Float32(v) => PublicAttributeValue::Float32(*v),
                    AttributeValue::Float32s(v) => PublicAttributeValue::Float32s(v.clone()),
                    AttributeValue::Int64(v) => PublicAttributeValue::Int64(*v),
                    AttributeValue::Int64s(v) => PublicAttributeValue::Int64s(v.clone()),
                    AttributeValue::String(v) => PublicAttributeValue::String(v.clone()),
                    AttributeValue::Strings(v) => PublicAttributeValue::Strings(v.clone()),
                    AttributeValue::Tensor(v) => PublicAttributeValue::Tensor(v.clone()),
                    AttributeValue::Tensors(v) => PublicAttributeValue::Tensors(v.clone()),
                    AttributeValue::DeferredGraph(_) | AttributeValue::Graph(_) => {
                        PublicAttributeValue::Graph
                    }
                    AttributeValue::DeferredGraphs(_) | AttributeValue::Graphs(_) => {
                        PublicAttributeValue::Graphs
                    }
                };
                (name.clone(), public)
            })
            .collect();
        Self(map)
    }

    /// The ONNX kind of an attribute, or `None` if absent.
    pub fn kind(&self, name: &str) -> Option<AttrKind> {
        Some(match self.0.get(name)? {
            PublicAttributeValue::Float32(_) => AttrKind::Float32,
            PublicAttributeValue::Float32s(_) => AttrKind::Float32s,
            PublicAttributeValue::Int64(_) => AttrKind::Int64,
            PublicAttributeValue::Int64s(_) => AttrKind::Int64s,
            PublicAttributeValue::String(_) => AttrKind::String,
            PublicAttributeValue::Strings(_) => AttrKind::Strings,
            PublicAttributeValue::Tensor(_) => AttrKind::Tensor,
            PublicAttributeValue::Tensors(_) => AttrKind::Tensors,
            PublicAttributeValue::Graph => AttrKind::Graph,
            PublicAttributeValue::Graphs => AttrKind::Graphs,
        })
    }

    /// Get an INT attribute.
    pub fn get_i64(&self, name: &str) -> Option<i64> {
        match self.0.get(name)? {
            PublicAttributeValue::Int64(v) => Some(*v),
            _ => None,
        }
    }

    /// Get an INTS attribute.
    pub fn get_i64s(&self, name: &str) -> Option<&[i64]> {
        match self.0.get(name)? {
            PublicAttributeValue::Int64s(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get a FLOAT attribute.
    pub fn get_f32(&self, name: &str) -> Option<f32> {
        match self.0.get(name)? {
            PublicAttributeValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a FLOATS attribute.
    pub fn get_f32s(&self, name: &str) -> Option<&[f32]> {
        match self.0.get(name)? {
            PublicAttributeValue::Float32s(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get a STRING attribute.
    pub fn get_string(&self, name: &str) -> Option<&str> {
        match self.0.get(name)? {
            PublicAttributeValue::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Get a STRINGS attribute.
    pub fn get_strings(&self, name: &str) -> Option<&[String]> {
        match self.0.get(name)? {
            PublicAttributeValue::Strings(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get a TENSOR attribute.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorData> {
        match self.0.get(name)? {
            PublicAttributeValue::Tensor(v) => Some(v),
            _ => None,
        }
    }

    /// Get a TENSORS attribute.
    pub fn get_tensors(&self, name: &str) -> Option<&[TensorData]> {
        match self.0.get(name)? {
            PublicAttributeValue::Tensors(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Iterate over the attribute names present on the node.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

impl AttributeValue {
    pub fn into_f32(self) -> f32 {
        if let AttributeValue::Float32(elem) = self {
            elem
        } else {
            panic!("Expected Float32, got {self:?}");
        }
    }

    pub fn into_i32(self) -> i32 {
        if let AttributeValue::Int64(elem) = self {
            elem as i32
        } else {
            panic!("Expected Int32, got {self:?}");
        }
    }

    pub fn into_i64(self) -> i64 {
        if let AttributeValue::Int64(elem) = self {
            elem
        } else {
            panic!("Expected Int64, got {self:?}");
        }
    }

    pub fn into_string(self) -> String {
        if let AttributeValue::String(elem) = self {
            elem
        } else {
            panic!("Expected String, got {self:?}");
        }
    }

    pub fn into_tensor(self) -> TensorData {
        if let AttributeValue::Tensor(elem) = self {
            elem
        } else {
            panic!("Expected Tensor, got {self:?}");
        }
    }

    #[allow(dead_code)]
    pub fn into_f32s(self) -> Vec<f32> {
        if let AttributeValue::Float32s(elem) = self {
            elem
        } else {
            panic!("Expected Float32s, got {self:?}");
        }
    }

    pub fn into_i64s(self) -> Vec<i64> {
        if let AttributeValue::Int64s(elem) = self {
            elem
        } else {
            panic!("Expected Int64s, got {self:?}");
        }
    }

    #[allow(dead_code)]
    pub fn into_strings(self) -> Vec<String> {
        if let AttributeValue::Strings(elem) = self {
            elem
        } else {
            panic!("Expected Strings, got {self:?}");
        }
    }

    #[allow(dead_code)]
    pub fn into_tensors(self) -> Vec<TensorData> {
        if let AttributeValue::Tensors(elem) = self {
            elem
        } else {
            panic!("Expected Tensors, got {self:?}");
        }
    }

    #[allow(dead_code)]
    pub fn into_graph(self) -> OnnxGraph {
        if let AttributeValue::Graph(elem) = self {
            elem
        } else {
            panic!("Expected Graph, got {self:?}");
        }
    }

    #[allow(dead_code)]
    pub fn into_graphs(self) -> Vec<OnnxGraph> {
        if let AttributeValue::Graphs(elem) = self {
            elem
        } else {
            panic!("Expected Graphs, got {self:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_fixture() -> Attributes {
        let mut attrs = Attributes::new();
        attrs.insert("n_fft".into(), AttributeValue::Int64(1024));
        attrs.insert("axes".into(), AttributeValue::Int64s(vec![0, 1]));
        attrs.insert("scale".into(), AttributeValue::Float32(0.5));
        attrs.insert("scales".into(), AttributeValue::Float32s(vec![1.0, 2.0]));
        attrs.insert("mode".into(), AttributeValue::String("real".into()));
        attrs.insert(
            "modes".into(),
            AttributeValue::Strings(vec!["a".into(), "b".into()]),
        );
        attrs.insert(
            "window".into(),
            AttributeValue::Tensor(TensorData::new(vec![0.25f32, 0.75], [2usize])),
        );
        attrs.insert(
            "windows".into(),
            AttributeValue::Tensors(vec![
                TensorData::new(vec![1i64], [1usize]),
                TensorData::new(vec![2i64, 3], [2usize]),
            ]),
        );
        attrs.insert(
            "body".into(),
            AttributeValue::DeferredGraph(DeferredGraph {
                proto: Arc::new(GraphProto::default()),
                opset_version: 16,
                domain_opsets: crate::pipeline::DomainOpsets::new(Default::default(), 16),
                name_registry: None,
                base_path: None,
                custom_op_inference: None,
            }),
        );
        attrs
    }

    #[test]
    fn scalar_accessors() {
        let attrs = PublicAttributesOwned::from_internal(&attrs_fixture());
        assert_eq!(attrs.get_i64("n_fft"), Some(1024));
        assert_eq!(attrs.get_f32("scale"), Some(0.5));
        assert_eq!(attrs.get_string("mode"), Some("real"));
    }

    #[test]
    fn list_accessors() {
        let attrs = PublicAttributesOwned::from_internal(&attrs_fixture());
        assert_eq!(attrs.get_i64s("axes"), Some(&[0i64, 1][..]));
        assert_eq!(attrs.get_f32s("scales"), Some(&[1.0f32, 2.0][..]));
        assert_eq!(
            attrs.get_strings("modes"),
            Some(&["a".to_string(), "b".to_string()][..])
        );
    }

    #[test]
    fn tensor_accessors() {
        let attrs = PublicAttributesOwned::from_internal(&attrs_fixture());
        let window = attrs.get_tensor("window").unwrap();
        assert_eq!(window.to_vec::<f32>().unwrap(), vec![0.25, 0.75]);
        let windows = attrs.get_tensors("windows").unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[1].to_vec::<i64>().unwrap(), vec![2, 3]);
    }

    #[test]
    fn graph_attributes_are_visible_but_payload_free() {
        let attrs = PublicAttributesOwned::from_internal(&attrs_fixture());
        // The graph attribute stays detectable via names() and kind()...
        let mut names: Vec<&str> = attrs.names().collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "axes", "body", "mode", "modes", "n_fft", "scale", "scales", "window", "windows"
            ]
        );
        assert_eq!(attrs.kind("body"), Some(AttrKind::Graph));
        // ...but no typed getter exposes a payload for it.
        assert_eq!(attrs.get_tensor("body"), None);
        assert_eq!(attrs.get_string("body"), None);
    }

    #[test]
    fn kind_distinguishes_absent_from_wrong_type() {
        let attrs = PublicAttributesOwned::from_internal(&attrs_fixture());
        // get_i64("scale") is None either way; kind() tells the difference.
        assert_eq!(attrs.get_i64("scale"), None);
        assert_eq!(attrs.kind("scale"), Some(AttrKind::Float32));
        assert_eq!(attrs.kind("nonexistent"), None);
        assert_eq!(attrs.kind("n_fft"), Some(AttrKind::Int64));
        assert_eq!(attrs.kind("windows"), Some(AttrKind::Tensors));
    }

    #[test]
    fn wrong_type_and_absent_lookups_return_none() {
        let attrs = PublicAttributesOwned::from_internal(&attrs_fixture());
        assert_eq!(attrs.get_i64("scale"), None);
        assert_eq!(attrs.get_i64s("n_fft"), None);
        assert_eq!(attrs.get_f32("n_fft"), None);
        assert_eq!(attrs.get_f32s("axes"), None);
        assert_eq!(attrs.get_string("window"), None);
        assert_eq!(attrs.get_strings("mode"), None);
        assert!(attrs.get_tensor("windows").is_none());
        assert!(attrs.get_tensors("window").is_none());
        assert_eq!(attrs.get_i64("nonexistent"), None);
    }
}
