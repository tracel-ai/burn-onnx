use super::{BurnImports, Scope, ToTokens};
use crate::LoadStrategy;
use crate::burn::custom_op::HookRegistry;
use crate::burn::node::NodeCodegen;
use crate::burn::node_codegen::{
    node_collect_snapshots, node_field, node_forward, node_register_imports,
};
use crate::burn::partition::{
    MIN_GRAPH_SIZE, Partition, reorder_constants_to_consumers, try_partition,
};
use burn_pack::{Tensor, Writer};
use burn_store::TensorSnapshot;
use onnx_ir::{Node, ir::ArgType};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::{collections::HashMap, path::PathBuf, sync::Arc};

/// Burn graph intermediate representation of modules and tensor operations.
#[derive(Debug)]
pub struct BurnGraph {
    nodes: Vec<Node>,
    scope: Scope,
    imports: BurnImports,
    top_comment: Option<String>,
    default: Option<TokenStream>,
    blank_spaces: bool,
    graph_input_args: Vec<onnx_ir::Argument>,
    graph_output_args: Vec<onnx_ir::Argument>,
    /// Whether to partition large graphs into submodules (default: true)
    partition: bool,
    /// Cached partition result (computed once, reused by snapshot collection and codegen)
    cached_partition: Option<Option<Partition>>,
    /// Graph I/O args that were converted from ScalarTensor to ScalarNative at the
    /// boundary. Maps arg name -> DType. Used to insert conversion code:
    /// - Outputs: `.into_scalar::<T>()` before the return
    /// - Inputs: `Tensor::from_data([name as T], &self.device)` after the params
    boundary_output_conversions: HashMap<String, onnx_ir::ir::DType>,
    boundary_input_conversions: HashMap<String, onnx_ir::ir::DType>,
    /// User codegen hooks for custom (non-built-in) ops
    hooks: Arc<HookRegistry>,
}

impl Default for BurnGraph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            scope: Scope::default(),
            imports: BurnImports::default(),
            top_comment: None,
            default: None,
            blank_spaces: false,
            graph_input_args: Vec::new(),
            graph_output_args: Vec::new(),
            partition: true,
            cached_partition: None,
            boundary_output_conversions: HashMap::new(),
            boundary_input_conversions: HashMap::new(),
            hooks: Arc::new(HookRegistry::default()),
        }
    }
}

impl BurnGraph {
    /// Register a new operation node into the graph.
    ///
    /// # Notes
    ///
    /// The node must be registered in the same order they will be executed in the forward pass.
    pub fn register(&mut self, node: Node) {
        log::debug!("Registering node => '{}'", node.name());
        self.nodes.push(node);
    }

    /// Save the state of each node in a burnpack file and generate weight-loading constructors.
    ///
    /// The [`LoadStrategy`] controls which constructors are generated on the `Model` struct.
    pub fn with_burnpack(mut self, out_file: PathBuf, strategy: LoadStrategy) -> Self {
        // Snapshot collection consults the hooks, so the subgraph guard must
        // run before it (codegen() re-checks for direct codegen users).
        self.validate_hooks_in_subgraphs();

        // Collect all tensor snapshots from nodes
        let snapshots = self.collect_all_snapshots();

        // Materialize snapshots into the tensor-agnostic burnpack representation.
        // This is currently eager because burn-pack doesn't accept lazy tensor providers.
        // See https://github.com/tracel-ai/burn/issues/5219.
        // FIXME: this is a regression that should be fixed for the official release
        let tensors = snapshots
            .iter()
            .map(|snapshot| {
                let data = snapshot.to_data().map_err(|e| (snapshot.full_path(), e))?;
                Ok(Tensor::new(
                    snapshot.full_path(),
                    snapshot.dtype,
                    snapshot.shape.clone(),
                    snapshot.tensor_id.map(|id| id.val()),
                    data.bytes,
                ))
            })
            .collect::<Result<Vec<_>, (String, burn_store::TensorSnapshotError)>>()
            .unwrap_or_else(|(path, e)| {
                panic!("Failed to materialize tensor snapshot {path}: {e}")
            });

        // Write burnpack file
        let burnpack_file = out_file.with_extension("bpk");
        Writer::new(tensors)
            .with_metadata("producer", "burn-onnx")
            .write_to_file(&burnpack_file)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to write burnpack file {}: {e}",
                    burnpack_file.display()
                )
            });

        // Register the loading code based on strategy
        if strategy != LoadStrategy::None {
            self.register_burnpack_loaders(burnpack_file, strategy);
        }

        self
    }

    /// Collect all tensor snapshots from nodes recursively.
    ///
    /// When partitioned into submodules, snapshot paths are prefixed with the submodule
    /// field name (e.g. "submodule1.linear1.weight") so that `load_from` routes weights
    /// to the correct nested module.
    fn collect_all_snapshots(&mut self) -> Vec<TensorSnapshot> {
        let partition = self.compute_partition();

        if let Some(partition) = partition {
            self.collect_snapshots_partitioned(&partition)
        } else {
            self.collect_snapshots_flat()
        }
    }

    /// Compute the partition once and cache it for reuse by both snapshot
    /// collection and codegen, avoiding redundant work and ensuring consistency.
    fn compute_partition(&mut self) -> Option<Partition> {
        if let Some(ref cached) = self.cached_partition {
            return cached.clone();
        }
        let result = if self.partition {
            // Move constants to just before their first consumer so they land
            // in the same chunk, avoiding wide forward() interfaces.
            // Only reorder for graphs large enough to actually partition.
            if self.nodes.len() >= MIN_GRAPH_SIZE {
                reorder_constants_to_consumers(&mut self.nodes);
            }
            try_partition(&self.nodes, &self.graph_input_args, &self.graph_output_args)
        } else {
            None
        };
        self.cached_partition = Some(result.clone());
        result
    }

    fn collect_snapshots_flat(&self) -> Vec<TensorSnapshot> {
        let mut snapshots = Vec::new();
        let mut field_name_counts: HashMap<String, usize> = HashMap::new();
        collect_snapshots_from_nodes(
            &self.nodes,
            "",
            &mut field_name_counts,
            &mut snapshots,
            &self.hooks,
        );
        snapshots
    }

    fn collect_snapshots_partitioned(&self, partition: &Partition) -> Vec<TensorSnapshot> {
        let mut snapshots = Vec::new();

        for (chunk_idx, range) in partition.chunks.iter().enumerate() {
            let prefix = format!("submodule{}", chunk_idx + 1);
            let chunk_nodes = &self.nodes[range.clone()];
            // Each chunk gets its own counter to match collect_fields_for_nodes (per-chunk)
            let mut field_name_counts: HashMap<String, usize> = HashMap::new();
            collect_snapshots_from_nodes(
                chunk_nodes,
                &prefix,
                &mut field_name_counts,
                &mut snapshots,
                &self.hooks,
            );
        }
        snapshots
    }

    /// Add blank spaces in some places
    ///
    /// # Notes
    ///
    /// It can be problematic when testing.
    pub fn with_blank_space(mut self, blank_spaces: bool) -> Self {
        self.blank_spaces = blank_spaces;
        self
    }

    /// Add a comment at the top of the generated file.
    pub fn with_top_comment(mut self, top_comment: Option<String>) -> Self {
        self.top_comment = top_comment;
        self
    }

    /// Enable or disable submodule partitioning for large models.
    pub fn with_partition(mut self, partition: bool) -> Self {
        self.partition = partition;
        self
    }

    /// Set the codegen hooks for custom (non-built-in) ops.
    ///
    /// Must be applied before `with_burnpack`, which collects snapshots and
    /// therefore consults the hooks for `Node::Custom` fields.
    pub(crate) fn with_hooks(mut self, hooks: Arc<HookRegistry>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Reject hook-relevant nodes inside If/Loop/Scan subgraph bodies.
    ///
    /// Subgraph body codegen dispatches through the hook-free `NodeCodegen`
    /// path (see `subgraph_helper`), so a custom op or an overridden built-in
    /// inside a body would silently emit the wrong code or fail to compile.
    /// Until subgraph codegen is hook-aware, fail up front with a message
    /// naming the node instead.
    fn validate_hooks_in_subgraphs(&self) {
        fn check_body(body: &onnx_ir::OnnxGraph, hooks: &HookRegistry) {
            for node in &body.nodes {
                if let Node::Custom(c) = node {
                    panic!(
                        "Custom op '{}' (node '{}') is inside an If/Loop/Scan body; \
                         custom op codegen inside subgraphs is not supported yet",
                        c, c.name
                    );
                }
                if self_override_matches(hooks, node) {
                    panic!(
                        "OpOverride for {:?} matches node '{}' inside an If/Loop/Scan body; \
                         overriding ops inside subgraphs is not supported yet",
                        node.node_type(),
                        node.name()
                    );
                }
                recurse(node, hooks);
            }
        }

        fn self_override_matches(hooks: &HookRegistry, node: &Node) -> bool {
            hooks.override_for(&node.node_type()).is_some()
        }

        fn recurse(node: &Node, hooks: &HookRegistry) {
            match node {
                Node::If(n) => {
                    check_body(&n.config.then_branch, hooks);
                    check_body(&n.config.else_branch, hooks);
                }
                Node::Loop(n) => check_body(&n.config.body, hooks),
                Node::Scan(n) => check_body(&n.config.body, hooks),
                _ => {}
            }
        }

        for node in &self.nodes {
            recurse(node, &self.hooks);
        }
    }

    /// Generate tokens representing the graph with Burn modules and tensor operations.
    pub fn codegen(mut self) -> TokenStream {
        self.validate_hooks_in_subgraphs();
        self.register_imports();

        let partition = self.compute_partition();

        if let Some(partition) = partition {
            self.codegen_partitioned(partition)
        } else {
            self.codegen_flat()
        }
    }

    /// Generate flat code (no submodules) for small graphs.
    fn codegen_flat(mut self) -> TokenStream {
        self.build_scope();

        let codegen_imports = self.imports.codegen();
        let codegen_struct = self.codegen_struct();
        let codegen_new = self.codegen_new();
        let codegen_forward = self.codegen_forward();

        let maybe_blank = match self.blank_spaces {
            true => quote! {
                _blank_!();
            },
            false => quote! {},
        };
        let codegen_default = match self.default {
            Some(default) => quote! {
                #default
                #maybe_blank
            },
            None => quote! {},
        };

        let maybe_top_file_comment = match self.top_comment {
            Some(comment) => quote! {
                _comment_!(#comment);
            },
            None => quote! {},
        };

        quote! {
            // @generated
            // This file is automatically generated by burn-onnx

            #maybe_top_file_comment
            #codegen_imports
            #maybe_blank
            #maybe_blank

            #codegen_struct
            #maybe_blank

            #codegen_default

            impl Model {
                #codegen_new

                #maybe_blank

                #codegen_forward
            }
        }
    }

    /// Generate partitioned code with submodule structs.
    fn codegen_partitioned(self, partition: Partition) -> TokenStream {
        let maybe_blank = match self.blank_spaces {
            true => quote! { _blank_!(); },
            false => quote! {},
        };

        let codegen_imports = self.imports.codegen();
        let maybe_top_file_comment = match &self.top_comment {
            Some(comment) => {
                let c = comment.clone();
                quote! { _comment_!(#c); }
            }
            None => quote! {},
        };

        let num_chunks = partition.chunks.len();
        let mut submodule_defs = Vec::with_capacity(num_chunks);
        let mut submodule_field_decls = Vec::with_capacity(num_chunks);
        let mut submodule_field_inits = Vec::with_capacity(num_chunks);
        let mut submodule_field_names = Vec::with_capacity(num_chunks);
        let mut forward_calls = Vec::with_capacity(num_chunks);

        // Count how many times each tensor is consumed across all chunk inputs.
        // This tells us when we need .clone() in the top-level forward.
        let mut remaining_uses: HashMap<String, usize> = HashMap::new();
        for inputs in &partition.chunk_inputs {
            for arg in inputs {
                *remaining_uses.entry(arg.name.clone()).or_insert(0) += 1;
            }
        }

        for (chunk_idx, range) in partition.chunks.iter().enumerate() {
            let struct_name = format_ident!("Submodule{}", chunk_idx + 1);
            let field_name = format_ident!("submodule{}", chunk_idx + 1);
            let chunk_nodes = &self.nodes[range.clone()];
            let chunk_inputs = &partition.chunk_inputs[chunk_idx];
            let chunk_outputs = &partition.chunk_outputs[chunk_idx];

            // Build scope for this chunk
            let mut scope = Scope::default();

            // Register chunk inputs as variables at position 0.
            // Mirror build_scope: also register boundary-converted inputs (ScalarNative
            // that were originally ScalarTensor) as tensor variables, since the top-level
            // forward converts them to Tensor<1> before calling submodule.forward().
            for arg in chunk_inputs {
                if matches!(arg.ty, ArgType::Tensor(_) | ArgType::ScalarTensor(_))
                    || self.boundary_input_conversions.contains_key(&arg.name)
                {
                    scope.tensor_register_variable(arg, 0);
                }
            }

            // Register node outputs and future uses (positions are local to this chunk)
            for (local_pos, node) in chunk_nodes.iter().enumerate() {
                for arg in node.outputs() {
                    if matches!(arg.ty, ArgType::Tensor(_) | ArgType::ScalarTensor(_)) {
                        scope.tensor_register_variable(arg, local_pos + 1);
                    }
                }
                for arg in node.inputs() {
                    if (arg.is_dynamic() || arg.is_constant())
                        && matches!(arg.ty, ArgType::Tensor(_) | ArgType::ScalarTensor(_))
                    {
                        scope.tensor_register_future_use(arg, local_pos);
                    }
                }
            }

            // Register chunk outputs as future uses at the end
            let chunk_len = chunk_nodes.len();
            for arg in chunk_outputs {
                if matches!(arg.ty, ArgType::Tensor(_) | ArgType::ScalarTensor(_)) {
                    scope.tensor_register_future_use(arg, chunk_len);
                }
            }

            // Collect fields from this chunk's nodes
            let chunk_fields = collect_fields_for_nodes(chunk_nodes, &self.hooks);

            // Generate the submodule struct body
            let struct_fields: Vec<_> = chunk_fields
                .iter()
                .map(|(name, ty, _)| quote! { #name: #ty, })
                .collect();

            // Generate new() body
            let field_init_code: TokenStream = chunk_fields
                .iter()
                .filter_map(|(_, _, init)| init.clone())
                .collect();
            let field_names_for_init: Vec<_> = chunk_fields
                .iter()
                .map(|(name, _, _)| name.clone())
                .collect();

            // Generate forward() body
            let input_params = crate::burn::codegen_fn_params(chunk_inputs);
            let output_type = crate::burn::codegen_return_type(chunk_outputs);
            let output_return = crate::burn::codegen_return_expr(chunk_outputs);

            let mut forward_body = quote! {};
            for (local_pos, node) in chunk_nodes.iter().enumerate() {
                let mut scope_at_pos = scope.at_position(local_pos);
                let code = node_forward(node, &mut scope_at_pos, &self.hooks);
                forward_body.extend(code);
            }

            let submodule_def = quote! {
                #[derive(Module, Debug)]
                pub struct #struct_name {
                    #(#struct_fields)*
                    #[module(skip)]
                    device: Device,
                }

                impl #struct_name {
                    #[allow(unused_variables)]
                    pub fn new(device: &Device) -> Self {
                        #field_init_code
                        Self {
                            #(#field_names_for_init,)*
                            device: device.clone(),
                        }
                    }

                    #[allow(clippy::let_and_return, clippy::approx_constant)]
                    pub fn forward(&self, #input_params) -> #output_type {
                        #forward_body
                        #output_return
                    }
                }
            };
            submodule_defs.push(submodule_def);

            // Top-level Model field for this submodule
            submodule_field_decls.push(quote! { #field_name: #struct_name, });
            submodule_field_inits.push(quote! { let #field_name = #struct_name::new(device); });
            submodule_field_names.push(field_name.clone());

            // Generate the forward call in the top-level forward().
            // Clone tensors that are consumed by later chunks.
            let input_args: Vec<_> = chunk_inputs
                .iter()
                .map(|arg| {
                    let name = crate::burn::arg_ident(arg);
                    let remaining = remaining_uses.get(&arg.name).copied().unwrap_or(0);
                    if remaining > 1 {
                        // Will be used again by a later chunk
                        remaining_uses.insert(arg.name.clone(), remaining - 1);
                        quote! { #name.clone() }
                    } else {
                        remaining_uses.remove(&arg.name);
                        quote! { #name }
                    }
                })
                .collect();

            if chunk_outputs.len() == 1 {
                let out_name = crate::burn::arg_ident(&chunk_outputs[0]);
                forward_calls.push(quote! {
                    let #out_name = self.#field_name.forward(#(#input_args),*);
                });
            } else {
                let out_names: Vec<_> = chunk_outputs.iter().map(crate::burn::arg_ident).collect();
                forward_calls.push(quote! {
                    let (#(#out_names),*) = self.#field_name.forward(#(#input_args),*);
                });
            }
        }

        // Top-level Model forward signature
        let input_def = crate::burn::codegen_fn_params(&self.graph_input_args);
        let output_type_def = crate::burn::codegen_return_type(&self.graph_output_args);
        let output_return_def = crate::burn::codegen_return_expr(&self.graph_output_args);

        let input_conversions = self.codegen_boundary_input_conversions();
        let boundary_conversions = self.codegen_boundary_output_conversions();

        let codegen_default = match &self.default {
            Some(default) => {
                let d = default.clone();
                quote! { #d #maybe_blank }
            }
            None => quote! {},
        };

        quote! {
            // @generated
            // This file is automatically generated by burn-onnx

            #maybe_top_file_comment
            #codegen_imports
            #maybe_blank
            #maybe_blank

            #(#submodule_defs)*
            #maybe_blank

            #[derive(Module, Debug)]
            pub struct Model {
                #(#submodule_field_decls)*
                #[module(skip)]
                device: Device,
            }
            #maybe_blank

            #codegen_default

            impl Model {
                #[allow(unused_variables)]
                pub fn new(device: &Device) -> Self {
                    #(#submodule_field_inits)*
                    Self {
                        #(#submodule_field_names,)*
                        device: device.clone(),
                    }
                }

                #maybe_blank

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, #input_def) -> #output_type_def {
                    #input_conversions
                    #(#forward_calls)*
                    #boundary_conversions
                    #output_return_def
                }
            }
        }
    }

    fn register_imports(&mut self) {
        // Register imports from nodes
        self.nodes
            .iter()
            .for_each(|node| node_register_imports(node, &mut self.imports, &self.hooks));
    }

    /// Build the scope state to make sure tensor clones are added where needed.
    fn build_scope(&mut self) {
        log::debug!("Building the scope nodes len => '{}'", self.nodes.len());

        // Register graph tensor inputs with 0 as node position
        self.graph_input_args
            .iter()
            .filter(|arg| {
                matches!(arg.ty, ArgType::Tensor(_) | ArgType::ScalarTensor(_))
                    || self.boundary_input_conversions.contains_key(&arg.name)
            })
            .for_each(|arg| {
                self.scope.tensor_register_variable(arg, 0);
            });

        self.nodes
            .iter()
            .enumerate()
            .for_each(|(node_position, node)| {
                // Register tensor outputs as variables
                node.outputs()
                    .iter()
                    .filter(|arg| matches!(arg.ty, ArgType::Tensor(_) | ArgType::ScalarTensor(_)))
                    .for_each(|arg| {
                        self.scope.tensor_register_variable(arg, node_position + 1);
                    });
                // Since the graph is guaranteed to be a DAG, we can safely register future uses
                // of the inputs (which are the previous nodes' outputs)
                // Filter to only dynamic/constant inputs (exclude static-only initializers)
                node.inputs()
                    .iter()
                    .filter(|arg| arg.is_dynamic() || arg.is_constant())
                    .filter(|arg| matches!(arg.ty, ArgType::Tensor(_) | ArgType::ScalarTensor(_)))
                    .for_each(|arg| self.scope.tensor_register_future_use(arg, node_position));
            });

        // Register graph tensor output with the last node position
        self.graph_output_args
            .iter()
            .filter(|arg| matches!(arg.ty, ArgType::Tensor(_) | ArgType::ScalarTensor(_)))
            .for_each(|arg| {
                self.scope.tensor_register_future_use(arg, self.nodes.len());
            });
    }

    fn register_burnpack_loaders(&mut self, file: PathBuf, strategy: LoadStrategy) {
        self.imports.register("burn_store::BurnpackStore");
        self.imports.register("burn_store::ModuleSnapshot");
        self.imports.register("burn::tensor::Bytes");

        let mut statics = quote! {};
        let mut default_impl = quote! {};
        let mut extra_loaders = quote! {};

        match strategy {
            LoadStrategy::File => {
                let file = path_to_str(&file);
                statics = quote! {
                    // `from_file` requires `std::path::Path`; opt into std so this
                    // also works when included from `#![no_std]` crates.
                    extern crate std;
                    _blank_!();
                };
                default_impl = quote! {
                    impl Default for Model {
                        fn default() -> Self {
                            Self::from_file(#file, &Default::default())
                        }
                    }
                    _blank_!();
                };
                extra_loaders = quote! {
                    /// Load model weights from a burnpack file.
                    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &Device) -> Self {
                        let mut model = Self::new(device);
                        let mut store = BurnpackStore::from_file(&file);
                        model.load_from(&mut store)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "Failed to load burnpack file {}: {e}",
                                    file.as_ref().display()
                                )
                            });
                        model
                    }
                    _blank_!();
                };
            }
            LoadStrategy::Embedded => {
                let file_size = std::fs::metadata(&file)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Failed to read burnpack file metadata {}: {e}",
                            file.display()
                        )
                    })
                    .len() as usize;
                let file = path_to_str(&file);
                statics = quote! {
                    // Align embedded data to 256-byte boundary to match burnpack's internal alignment.
                    // This ensures tensor data remains properly aligned for zero-copy loading,
                    // regardless of where the linker places the static data in the binary.
                    #[repr(C, align(256))]
                    struct Aligned256([u8; #file_size]);
                    static ALIGNED_DATA: Aligned256 = Aligned256(*include_bytes!(#file));
                    static EMBEDDED_STATES: &[u8] = &ALIGNED_DATA.0;
                    _blank_!();
                };
                default_impl = quote! {
                    impl Default for Model {
                        fn default() -> Self {
                            Self::from_embedded(&Default::default())
                        }
                    }
                    _blank_!();
                };
                extra_loaders = quote! {
                    /// Load model weights from embedded burnpack data (zero-copy at store level).
                    ///
                    /// The embedded data stays in the binary's .rodata section without heap allocation.
                    /// Tensor data is sliced directly from the static bytes.
                    ///
                    /// Note: Some backends may still copy data internally.
                    /// See <https://github.com/tracel-ai/burn/issues/4153> for true backend zero-copy.
                    ///
                    /// See <https://github.com/tracel-ai/burn/issues/4123>
                    pub fn from_embedded(device: &Device) -> Self {
                        let mut model = Self::new(device);
                        let mut store = BurnpackStore::from_static(EMBEDDED_STATES);
                        model.load_from(&mut store)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "Failed to load embedded burnpack (built from {}): {e}",
                                    #file
                                )
                            });
                        model
                    }
                    _blank_!();
                };
            }
            LoadStrategy::Bytes | LoadStrategy::None => {}
        }

        self.default = Some(quote! {
            _blank_!();
            #statics
            #default_impl
            impl Model {
                #extra_loaders
                /// Load model weights from in-memory bytes.
                ///
                /// The bytes must be the contents of a `.bpk` file.
                pub fn from_bytes(bytes: Bytes, device: &Device) -> Self {
                    let mut model = Self::new(device);
                    let mut store = BurnpackStore::from_bytes(Some(bytes));
                    model.load_from(&mut store)
                        .unwrap_or_else(|e| panic!("Failed to load burnpack bytes: {e}"));
                    model
                }
            }
        });
    }

    /// Recursively collect all fields from nodes, including subgraph nodes in If/Loop/Scan
    fn collect_all_fields(&self) -> Vec<FieldTuple> {
        collect_fields_for_nodes(&self.nodes, &self.hooks)
    }

    fn codegen_struct(&self) -> TokenStream {
        let mut body = quote! {};
        self.collect_all_fields()
            .iter()
            .map(|(name, ty, _)| {
                quote! {
                    #name: #ty,
                }
            })
            .for_each(|code| body.extend(code));

        body.extend(quote! {
            #[module(skip)]
            device: Device,
        });

        quote! {
            #[derive(Module, Debug)]
            pub struct Model {
                #body
            }
        }
    }

    fn codegen_new(&self) -> TokenStream {
        let mut body = quote! {};
        let all_fields = self.collect_all_fields();

        // Generate field initialization code
        for (_, _, field_init) in &all_fields {
            body.extend(field_init.clone());
        }

        // Collect field names for struct initialization
        let field_names: Vec<_> = all_fields.iter().map(|(name, _, _)| name.clone()).collect();

        quote! {
            #[allow(unused_variables)]
            pub fn new(device: &Device) -> Self {
                #body

                Self {
                    #(#field_names,)*
                    device: device.clone(),
                }
            }
        }
    }

    fn codegen_forward(&mut self) -> TokenStream {
        let input_def = crate::burn::codegen_fn_params(&self.graph_input_args);
        let output_type_def = crate::burn::codegen_return_type(&self.graph_output_args);
        let output_return_def = crate::burn::codegen_return_expr(&self.graph_output_args);

        let input_conversions = self.codegen_boundary_input_conversions();

        let mut body = quote! {};
        for (index, node) in self.nodes.iter().enumerate() {
            let mut scope_at_pos = self.scope.at_position(index);
            let code = node_forward(node, &mut scope_at_pos, &self.hooks);
            body.extend(code);
        }

        let boundary_conversions = self.codegen_boundary_output_conversions();

        // TODO Return the result without a `let` binding from a block,
        // otherwise let_and_return error will be triggered by clippy.
        // For now, we just disable the warning.
        quote! {
            #[allow(clippy::let_and_return, clippy::approx_constant)]
            pub fn forward(&self, #input_def) -> #output_type_def {
                #input_conversions
                #body
                #boundary_conversions
                #output_return_def
            }
        }
    }

    /// Register the input and output types of the graph using the passed in names.
    /// The names must be unique and match the names of the inputs and outputs of the nodes.
    /// The order will be preserved.
    ///
    /// # Arguments
    ///
    /// * `input_names` - The names of the inputs of the graph.
    /// * `output_names` - The names of the outputs of the graph.
    /// * `input_args` - The input arguments (from ONNX graph, used for empty graphs).
    /// * `output_args` - The output arguments (from ONNX graph, used for empty graphs).
    pub fn register_input_output(
        &mut self,
        input_names: Vec<String>,
        output_names: Vec<String>,
        input_args: &[onnx_ir::Argument],
        output_args: &[onnx_ir::Argument],
    ) {
        // Handle empty graphs: use provided arguments directly
        if self.nodes.is_empty() {
            // For empty graphs, inputs pass through directly to outputs
            self.graph_input_args.extend_from_slice(input_args);
            self.graph_output_args.extend_from_slice(output_args);
            self.convert_graph_boundary_scalars();
            return;
        }

        // Get the unique names of each input/output of the nodes
        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        for node in self.nodes.iter() {
            for input_arg in NodeCodegen::inputs(node) {
                inputs.insert(input_arg.name.clone(), input_arg.clone());
            }
            for output_arg in NodeCodegen::outputs(node) {
                outputs.insert(output_arg.name.clone(), output_arg.clone());
            }
        }

        // Get the input arguments of the graph using passed in names
        // For outer scope variables, fall back to the provided input_args
        input_names.iter().enumerate().for_each(|(idx, input)| {
            let input_arg = inputs
                .get(input)
                .cloned()
                .or_else(|| {
                    // Fall back to provided input_args for outer scope variables
                    if idx < input_args.len() {
                        Some(input_args[idx].clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| panic!("Input argument not found for {input}"));

            self.graph_input_args.push(input_arg);
        });

        // Handle outputs - if output_args is provided (from ONNX), use it with renaming
        // Otherwise, look up arguments from node outputs (for tests)
        if !output_args.is_empty() {
            output_names
                .iter()
                .zip(output_args.iter())
                .for_each(|(name, arg)| {
                    // Rename argument to the graph output name
                    let mut renamed_arg = arg.clone();
                    renamed_arg.name = name.clone();
                    self.graph_output_args.push(renamed_arg);
                });
        } else {
            // For tests and non-ONNX usage: look up arguments from node outputs
            output_names.iter().for_each(|output| {
                self.graph_output_args.push(
                    outputs
                        .get(output)
                        .unwrap_or_else(|| panic!("Output argument not found for {output}"))
                        .clone(),
                );
            });
        }

        // Convert ScalarTensor to ScalarNative at graph boundary so user-facing
        // forward() signatures use native types (f32, i64, etc.) not Tensor<1>
        self.convert_graph_boundary_scalars();
    }

    /// Generate ScalarNative -> ScalarTensor input conversion code for graph boundary.
    fn codegen_boundary_input_conversions(&self) -> TokenStream {
        let mut tokens = quote! {};
        for arg in &self.graph_input_args {
            if let Some(dtype) = self.boundary_input_conversions.get(&arg.name) {
                let name = crate::burn::arg_ident(arg);
                let dtype_tokens = dtype.to_tokens();
                if dtype.is_float() {
                    tokens.extend(quote! {
                        let #name = Tensor::<1>::from_data(
                            burn::tensor::TensorData::from([#name]),
                            (&self.device, #dtype_tokens)
                        );
                    });
                } else if dtype.is_int() || dtype.is_uint() {
                    tokens.extend(quote! {
                        let #name = Tensor::<1, Int>::from_data(
                            burn::tensor::TensorData::from([#name]),
                            (&self.device, #dtype_tokens)
                        );
                    });
                } else if dtype.is_bool() {
                    tokens.extend(quote! {
                        let #name = Tensor::<1, Bool>::from_data(
                            burn::tensor::TensorData::from([#name]),
                            (&self.device, #dtype_tokens)
                        );
                    });
                } else {
                    panic!(
                        "Unsupported dtype {:?} for graph boundary ScalarNative -> ScalarTensor conversion",
                        dtype
                    );
                }
            }
        }
        tokens
    }

    /// Generate ScalarTensor -> ScalarNative output conversion code for graph boundary.
    fn codegen_boundary_output_conversions(&self) -> TokenStream {
        let mut tokens = quote! {};
        for arg in &self.graph_output_args {
            if let Some(dtype) = self.boundary_output_conversions.get(&arg.name) {
                let name = crate::burn::arg_ident(arg);
                let convert = crate::burn::on_device_to_native(quote! { #name }, dtype);
                tokens.extend(quote! {
                    let #name = #convert;
                });
            }
        }
        tokens
    }

    /// Convert ScalarTensor to ScalarNative at graph I/O boundary.
    /// Users pass/receive native scalars; internal representation is on-device.
    fn convert_graph_boundary_scalars(&mut self) {
        for arg in &mut self.graph_input_args {
            if let ArgType::ScalarTensor(dtype) = arg.ty {
                self.boundary_input_conversions
                    .insert(arg.name.clone(), dtype);
                arg.ty = ArgType::ScalarNative(dtype);
            }
        }
        for arg in &mut self.graph_output_args {
            if let ArgType::ScalarTensor(dtype) = arg.ty {
                self.boundary_output_conversions
                    .insert(arg.name.clone(), dtype);
                arg.ty = ArgType::ScalarNative(dtype);
            }
        }
    }
}

// ============================================================================
// Free functions shared by flat and partitioned codegen paths
// ============================================================================

type FieldTuple = (proc_macro2::Ident, TokenStream, Option<TokenStream>);

/// Render a burnpack path as `&str` for embedding into generated source.
///
/// The path is baked into the generated code as a string literal (`from_file(#file)`,
/// `include_bytes!(#file)`), so a non-UTF-8 path cannot be represented at all.
fn path_to_str(path: &std::path::Path) -> &str {
    path.to_str().unwrap_or_else(|| {
        panic!(
            "Burnpack path is not valid UTF-8 and cannot be embedded in generated code: {}",
            path.display()
        )
    })
}

/// Collect fields from a slice of nodes (including If/Loop subgraph fields).
fn collect_fields_for_nodes(nodes: &[Node], hooks: &HookRegistry) -> Vec<FieldTuple> {
    let mut field_name_counts: HashMap<String, usize> = HashMap::new();
    let mut all_fields: Vec<FieldTuple> = Vec::new();

    // Subgraph bodies are deliberately hook-FREE: their forward codegen goes
    // through the built-in NodeCodegen path (subgraph_helper), and
    // validate_hooks_in_subgraphs rejects any body node a hook would affect.
    // Consulting hooks here while forward does not would desynchronize
    // fields from the emitted code.
    fn collect_subgraph_fields_recursive(
        subgraph: &onnx_ir::OnnxGraph,
        field_name_counts: &mut HashMap<String, usize>,
        all_fields: &mut Vec<FieldTuple>,
    ) {
        for node in &subgraph.nodes {
            if let Some(mut field) = NodeCodegen::field(node) {
                let base_name = field.name.to_string();
                let count = field_name_counts.entry(base_name.clone()).or_insert(0);
                *count += 1;

                if *count > 1 {
                    let new_name_str = format!("{}_{}", base_name, count);
                    let new_name = syn::Ident::new(&new_name_str, proc_macro2::Span::call_site());
                    field.name = new_name;

                    let init_str = field.init.to_string();
                    let updated = init_str
                        .replace(
                            &format!("let {} :", base_name),
                            &format!("let {} :", new_name_str),
                        )
                        .replace(
                            &format!("let {} =", base_name),
                            &format!("let {} =", new_name_str),
                        );
                    field.init = updated.parse().unwrap_or_else(|e| {
                        log::warn!(
                            "Failed to parse renamed field init for '{}': {e}",
                            new_name_str
                        );
                        field.init.clone()
                    });
                }
                all_fields.push((field.name.clone(), field.ty.clone(), Some(field.init)));
            }

            if let Node::If(nested) = node {
                collect_subgraph_fields_recursive(
                    &nested.config.then_branch,
                    field_name_counts,
                    all_fields,
                );
                collect_subgraph_fields_recursive(
                    &nested.config.else_branch,
                    field_name_counts,
                    all_fields,
                );
            } else if let Node::Loop(nested) = node {
                collect_subgraph_fields_recursive(
                    &nested.config.body,
                    field_name_counts,
                    all_fields,
                );
            }
        }
    }

    for node in nodes {
        if let Some(field) = node_field(node, hooks) {
            all_fields.push((field.name, field.ty, Some(field.init)));
        }

        if let Node::If(if_node) = node {
            collect_subgraph_fields_recursive(
                &if_node.config.then_branch,
                &mut field_name_counts,
                &mut all_fields,
            );
            collect_subgraph_fields_recursive(
                &if_node.config.else_branch,
                &mut field_name_counts,
                &mut all_fields,
            );
        } else if let Node::Loop(loop_node) = node {
            collect_subgraph_fields_recursive(
                &loop_node.config.body,
                &mut field_name_counts,
                &mut all_fields,
            );
        }
    }

    all_fields
}

/// Collect tensor snapshots from a slice of nodes, optionally prefixing paths.
///
/// When `prefix` is non-empty, snapshot paths become "prefix.field.weight" etc.
fn collect_snapshots_from_nodes(
    nodes: &[Node],
    prefix: &str,
    field_name_counts: &mut HashMap<String, usize>,
    snapshots: &mut Vec<TensorSnapshot>,
    hooks: &HookRegistry,
) {
    // Hook-free for the same reason as collect_subgraph_fields_recursive:
    // subgraph forward codegen is hook-free, and hook-relevant body nodes
    // are rejected up front.
    fn collect_subgraph_snapshots_recursive(
        subgraph: &onnx_ir::OnnxGraph,
        prefix: &str,
        field_name_counts: &mut HashMap<String, usize>,
        snapshots: &mut Vec<TensorSnapshot>,
    ) {
        for node in &subgraph.nodes {
            if let Some(field) = NodeCodegen::field(node) {
                let base_name = field.name.to_string();
                let count = field_name_counts.entry(base_name.clone()).or_insert(0);
                *count += 1;

                let unique_name = if *count > 1 {
                    format!("{}_{}", base_name, count)
                } else {
                    base_name
                };

                let full_name = if prefix.is_empty() {
                    unique_name
                } else {
                    format!("{}.{}", prefix, unique_name)
                };
                let node_snapshots = NodeCodegen::collect_snapshots(node, &full_name);
                snapshots.extend(node_snapshots);
            }

            if let Node::If(nested) = node {
                collect_subgraph_snapshots_recursive(
                    &nested.config.then_branch,
                    prefix,
                    field_name_counts,
                    snapshots,
                );
                collect_subgraph_snapshots_recursive(
                    &nested.config.else_branch,
                    prefix,
                    field_name_counts,
                    snapshots,
                );
            } else if let Node::Loop(nested) = node {
                collect_subgraph_snapshots_recursive(
                    &nested.config.body,
                    prefix,
                    field_name_counts,
                    snapshots,
                );
            }
        }
    }

    for node in nodes {
        if let Some(field) = node_field(node, hooks) {
            let base_name = field.name.to_string();
            let count = field_name_counts.entry(base_name.clone()).or_insert(0);
            *count += 1;

            let unique_name = if *count > 1 {
                format!("{}_{}", base_name, count)
            } else {
                base_name
            };

            let full_name = if prefix.is_empty() {
                unique_name
            } else {
                format!("{}.{}", prefix, unique_name)
            };
            let node_snapshots = node_collect_snapshots(node, &full_name, hooks);
            snapshots.extend(node_snapshots);
        }

        if let Node::If(if_node) = node {
            collect_subgraph_snapshots_recursive(
                &if_node.config.then_branch,
                prefix,
                field_name_counts,
                snapshots,
            );
            collect_subgraph_snapshots_recursive(
                &if_node.config.else_branch,
                prefix,
                field_name_counts,
                snapshots,
            );
        } else if let Node::Loop(loop_node) = node {
            collect_subgraph_snapshots_recursive(
                &loop_node.config.body,
                prefix,
                field_name_counts,
                snapshots,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::DType;
    use onnx_ir::node::abs::AbsNodeBuilder;
    use rust_format::{Config, Formatter, PostProcess, PrettyPlease};

    fn format_tokens(tokens: TokenStream) -> String {
        let config = Config::new_str().post_proc(PostProcess::ReplaceMarkersAndDocBlocks);
        let formatter = PrettyPlease::from_config(config);
        formatter
            .format_tokens(tokens)
            .unwrap_or_else(|_| "FORMATTING FAILED".to_string())
    }

    /// input -> FftLike (custom, domain my.domain) -> output
    fn build_custom_op_graph() -> BurnGraph {
        use onnx_ir::ir::TensorType;

        let mut graph = BurnGraph::default();

        let custom = onnx_ir::CustomNode::new(
            "fftlike1",
            "FftLike",
            "my.domain",
            vec![onnx_ir::Argument::new(
                "input",
                ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
            )],
            vec![onnx_ir::Argument::new(
                "t0",
                ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
            )],
            Default::default(),
            3,
        );
        graph.register(Node::Custom(custom));

        graph.register_input_output(vec!["input".to_string()], vec!["t0".to_string()], &[], &[]);

        graph
    }

    /// Hook that emits a call into a user crate for FftLike nodes.
    struct FftLikeOp;

    impl crate::ext::CustomOp for FftLikeOp {
        fn op_type(&self) -> &str {
            "FftLike"
        }

        fn domain(&self) -> &str {
            "my.domain"
        }

        fn infer_output_types(
            &self,
            node: &onnx_ir::CustomNode,
        ) -> Result<Vec<ArgType>, onnx_ir::ProcessError> {
            Ok(vec![node.inputs[0].ty.clone()])
        }

        fn forward(
            &self,
            node: &onnx_ir::CustomNode,
            ctx: &mut crate::ext::CodegenContext<'_, '_>,
        ) -> Result<TokenStream, onnx_ir::ProcessError> {
            let input = ctx.arg(&node.inputs[0]);
            let out = crate::burn::node_traits::arg_to_ident(&node.outputs[0]);
            Ok(quote! {
                let #out = my_crate::ops::fft_like(#input);
            })
        }

        fn register_imports(&self, imports: &mut crate::ext::Imports<'_>) {
            imports.register("my_crate::ops");
        }
    }

    #[test]
    fn custom_op_codegen_dispatches_to_hook() {
        let mut registry = HookRegistry::default();
        registry.add_custom_op(Box::new(FftLikeOp));

        let graph = build_custom_op_graph().with_hooks(Arc::new(registry));
        let code = format_tokens(graph.codegen());

        // Hook-emitted forward line and hook-registered import
        assert!(
            code.contains("let t0 = my_crate::ops::fft_like(input);"),
            "generated code missing hook output:\n{code}"
        );
        assert!(
            code.contains("use my_crate::ops;"),
            "generated code missing hook import:\n{code}"
        );
    }

    #[test]
    #[should_panic(expected = "has no registered hook")]
    fn custom_op_without_hook_panics_with_clear_message() {
        let graph = build_custom_op_graph();
        graph.codegen();
    }

    /// Custom op that declares a module field and a weight snapshot,
    /// exercising the node_field / node_collect_snapshots dispatch.
    struct StatefulFftOp;

    impl crate::ext::CustomOp for StatefulFftOp {
        fn op_type(&self) -> &str {
            "FftLike"
        }

        fn domain(&self) -> &str {
            "my.domain"
        }

        fn infer_output_types(
            &self,
            node: &onnx_ir::CustomNode,
        ) -> Result<Vec<ArgType>, onnx_ir::ProcessError> {
            Ok(vec![node.inputs[0].ty.clone()])
        }

        fn forward(
            &self,
            node: &onnx_ir::CustomNode,
            ctx: &mut crate::ext::CodegenContext<'_, '_>,
        ) -> Result<TokenStream, onnx_ir::ProcessError> {
            let input = ctx.arg(&node.inputs[0]);
            let out = crate::burn::node_traits::arg_to_ident(&node.outputs[0]);
            Ok(quote! {
                let #out = my_crate::ops::stateful_fft(#input, self.fft_state.val());
            })
        }

        fn field(
            &self,
            _node: &onnx_ir::CustomNode,
        ) -> Result<Option<crate::burn::Field>, onnx_ir::ProcessError> {
            Ok(Some(crate::burn::Field::new(
                "fft_state",
                quote! { burn::module::Param<Tensor<1>> },
                quote! {
                    let fft_state = burn::module::Param::from_tensor(
                        Tensor::zeros([4], device),
                    );
                },
            )))
        }

        fn collect_snapshots(
            &self,
            _node: &onnx_ir::CustomNode,
            field_name: &str,
        ) -> Result<Vec<TensorSnapshot>, onnx_ir::ProcessError> {
            use burn::module::ParamId;
            use burn::tensor::TensorData;
            let data_fn =
                std::rc::Rc::new(|| Ok(TensorData::new(vec![0.5f32, 1.0, 1.5, 2.0], [4usize])));
            Ok(vec![TensorSnapshot::from_closure(
                data_fn,
                burn::tensor::DType::F32,
                [4usize].into(),
                vec![field_name.to_string(), "state".to_string()],
                vec!["Struct:StatefulFft".to_string()],
                ParamId::new(),
            )])
        }
    }

    #[test]
    fn custom_op_field_and_snapshots_dispatch_to_hook() {
        let mut registry = HookRegistry::default();
        registry.add_custom_op(Box::new(StatefulFftOp));
        let registry = Arc::new(registry);

        // Snapshot collection consults the hook's field and collect_snapshots
        let graph = build_custom_op_graph().with_hooks(registry.clone());
        let node = &graph.nodes[0];
        let field =
            crate::burn::node_codegen::node_field(node, &registry).expect("hook-declared field");
        assert_eq!(field.name.to_string(), "fft_state");
        let snapshots =
            crate::burn::node_codegen::node_collect_snapshots(node, "fft_state", &registry);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].full_path(), "fft_state.state");

        // The generated struct and forward reference the hook's field
        let code = format_tokens(graph.codegen());
        assert!(
            code.contains("fft_state: burn::module::Param<Tensor<1>>"),
            "struct field missing:\n{code}"
        );
        assert!(
            code.contains("self.fft_state.val()"),
            "forward does not use the field:\n{code}"
        );
    }

    /// Parse a committed fixture (shared with onnx-ir's integration tests)
    /// and wire it into a BurnGraph the way ModelGen does.
    fn build_graph_from_fixture(name: &str) -> BurnGraph {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../onnx-ir/tests/fixtures")
            .join(name);
        let parsed = onnx_ir::OnnxGraphBuilder::new()
            .simplify(false)
            .parse_file(&path)
            .expect("fixture should parse");

        let mut graph = BurnGraph::default();
        let input_names: Vec<_> = parsed.inputs.iter().map(|a| a.name.clone()).collect();
        let output_names: Vec<_> = parsed.outputs.iter().map(|a| a.name.clone()).collect();
        graph.register_input_output(input_names, output_names, &parsed.inputs, &parsed.outputs);
        for node in parsed.nodes {
            graph.register(node);
        }
        graph
    }

    #[test]
    #[should_panic(expected = "inside an If/Loop/Scan body")]
    fn custom_op_in_subgraph_is_rejected_at_codegen() {
        build_graph_from_fixture("custom_in_if.onnx").codegen();
    }

    /// Override stub targeting Relu, for the subgraph rejection test.
    struct ReluOverrideStub;

    impl crate::ext::OpOverride for ReluOverrideStub {
        fn target(&self) -> onnx_ir::NodeType {
            onnx_ir::NodeType::Relu
        }

        fn forward(
            &self,
            _node: &Node,
            _ctx: &mut crate::ext::CodegenContext<'_, '_>,
        ) -> Result<TokenStream, onnx_ir::ProcessError> {
            Ok(TokenStream::new())
        }
    }

    #[test]
    #[should_panic(expected = "inside an If/Loop/Scan body")]
    fn override_target_in_subgraph_is_rejected_at_codegen() {
        let mut registry = HookRegistry::default();
        registry.add_override(Box::new(ReluOverrideStub));
        build_graph_from_fixture("relu_in_if.onnx")
            .with_hooks(Arc::new(registry))
            .codegen();
    }

    /// Override that reroutes built-in Abs through a user kernel.
    struct AbsOverride;

    impl crate::ext::OpOverride for AbsOverride {
        fn target(&self) -> onnx_ir::NodeType {
            onnx_ir::NodeType::Abs
        }

        fn forward(
            &self,
            node: &Node,
            ctx: &mut crate::ext::CodegenContext<'_, '_>,
        ) -> Result<TokenStream, onnx_ir::ProcessError> {
            let Node::Abs(abs) = node else {
                panic!("expected Abs node");
            };
            let input = ctx.arg(&abs.inputs[0]);
            let out = crate::burn::node_traits::arg_to_ident(&abs.outputs[0]);
            Ok(quote! {
                let #out = my_crate::kernels::fast_abs(#input);
            })
        }

        fn register_imports(&self, imports: &mut crate::ext::Imports<'_>) {
            imports.register("my_crate::kernels");
        }
    }

    #[test]
    fn op_override_replaces_builtin_codegen() {
        let mut registry = HookRegistry::default();
        registry.add_override(Box::new(AbsOverride));

        let graph = build_abs_chain(2).with_hooks(Arc::new(registry));
        let code = format_tokens(graph.codegen());

        assert!(
            code.contains("let t0 = my_crate::kernels::fast_abs(input);"),
            "override output missing:\n{code}"
        );
        assert!(
            code.contains("let t1 = my_crate::kernels::fast_abs(t0);"),
            "override output missing for second node:\n{code}"
        );
        assert!(
            code.contains("use my_crate::kernels;"),
            "override import missing:\n{code}"
        );
        // The built-in Abs codegen must not appear anywhere
        assert!(!code.contains(".abs()"), "builtin abs leaked:\n{code}");
    }

    #[test]
    fn builtin_codegen_unchanged_without_override() {
        let graph = build_abs_chain(1);
        let code = format_tokens(graph.codegen());
        assert!(code.contains(".abs()"), "builtin abs missing:\n{code}");
    }

    /// Build a chain of N abs nodes: input -> t0 -> t1 -> ... -> t{N-1}
    fn build_abs_chain(n: usize) -> BurnGraph {
        let mut graph = BurnGraph::default();

        for i in 0..n {
            let in_name = if i == 0 {
                "input".to_string()
            } else {
                format!("t{}", i - 1)
            };
            let out_name = format!("t{}", i);

            let node = AbsNodeBuilder::new(format!("abs{}", i))
                .input_tensor(&in_name, 2, DType::F32)
                .output_tensor(&out_name, 2, DType::F32)
                .build();

            graph.register(Node::Abs(node));
        }

        let last_out = format!("t{}", n - 1);
        graph.register_input_output(vec!["input".to_string()], vec![last_out], &[], &[]);

        graph
    }

    /// Two Clip nodes chained through a single intermediate tensor,
    /// each with its own independent runtime scalar bounds. The
    /// generated `__clip_min` / `__clip_max` temporaries must each
    /// live inside their own per-node block so clone-tracking and
    /// name resolution don't interleave across the two instances.
    fn build_two_clip_chain() -> BurnGraph {
        use onnx_ir::clip::{ClipConfig, ClipNodeBuilder};
        use onnx_ir::node::clip::ClipInput;

        let mut graph = BurnGraph::default();

        let mk = |name: &str, in_tensor: &str, min_name: &str, max_name: &str, out: &str| {
            ClipNodeBuilder::new(name)
                .input_tensor(in_tensor, 2, DType::F32)
                .input_scalar(min_name, DType::F32)
                .input_scalar(max_name, DType::F32)
                .output_tensor(out, 2, DType::F32)
                .config(ClipConfig {
                    min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                        min_name.to_string(),
                        1,
                    ))),
                    max: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                        max_name.to_string(),
                        2,
                    ))),
                })
                .build()
        };

        graph.register(Node::Clip(mk("clip0", "input", "min0", "max0", "t0")));
        graph.register(Node::Clip(mk("clip1", "t0", "min1", "max1", "t1")));

        graph.register_input_output(
            vec![
                "input".to_string(),
                "min0".to_string(),
                "max0".to_string(),
                "min1".to_string(),
                "max1".to_string(),
            ],
            vec!["t1".to_string()],
            &[],
            &[],
        );

        graph
    }

    /// Walk the generated Rust text and return the list of innermost
    /// `{ ... }` blocks (as substrings of `code`, without the braces).
    /// Used by scoping regression tests: counting raw occurrences of a
    /// `let __foo` binding is not enough because the same bindings at
    /// the outer `forward` scope would still pass. An innermost-block
    /// scan lets us assert that the bindings sit inside a per-node
    /// subscope, not at the function top level.
    ///
    /// "Innermost" means the block contains no nested `{...}` children.
    /// Tracked per-block on the stack so siblings don't pollute each
    /// other (a parent with one inner child is still a parent — not
    /// innermost — but its other children can still qualify).
    fn innermost_blocks(code: &str) -> Vec<&str> {
        let bytes = code.as_bytes();
        let mut stack: Vec<(usize, bool)> = Vec::new();
        let mut innermost: Vec<(usize, usize)> = Vec::new();
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'{' => {
                    if let Some(last) = stack.last_mut() {
                        last.1 = true;
                    }
                    stack.push((i, false));
                }
                b'}' => {
                    if let Some((open, has_inner)) = stack.pop()
                        && !has_inner
                    {
                        innermost.push((open + 1, i));
                    }
                }
                _ => {}
            }
        }
        innermost.into_iter().map(|(s, e)| &code[s..e]).collect()
    }

    /// Regression test for #317, issue 6: verifies that runtime-bound Clip
    /// nodes emit their `__clip_min` / `__clip_max` temporaries inside
    /// per-node block scopes rather than at the outer `forward` scope.
    /// Without the wrapper block, both `let __clip_min = ...;` bindings
    /// would land at the outer scope — still legal Rust, but
    /// clone-tracking for the runtime-bound inputs and variable
    /// resolution for downstream consumers would interleave across nodes
    /// in hard-to-debug ways.
    ///
    /// The test walks the generated code to find every innermost `{ ... }`
    /// block and counts the ones that contain both a `let __clip_min = `
    /// and a `let __clip_max = `. That count must be exactly two (one per
    /// Clip node). A raw `code.matches(...).count() == 2` would also pass
    /// if both bindings were at the outer scope, which is exactly the
    /// regression we are trying to rule out.
    #[test]
    fn multi_instance_clip_scoping() {
        let graph = build_two_clip_chain();
        let code = format_tokens(graph.codegen());

        let scoped_blocks: Vec<&str> = innermost_blocks(&code)
            .into_iter()
            .filter(|b| b.contains("let __clip_min = ") && b.contains("let __clip_max = "))
            .collect();

        assert_eq!(
            scoped_blocks.len(),
            2,
            "expected exactly two innermost blocks each containing \
             both `let __clip_min =` and `let __clip_max =`, got \
             {} such blocks. Full generated code:\n{code}",
            scoped_blocks.len()
        );

        // Belt-and-braces: each scoped block must declare exactly one
        // `__clip_min` and one `__clip_max`. A block containing two
        // `__clip_min` bindings would mean two clip nodes collapsed into
        // a single scope, which is the bug we are guarding against.
        for (idx, block) in scoped_blocks.iter().enumerate() {
            assert_eq!(
                block.matches("let __clip_min = ").count(),
                1,
                "block {idx} should declare exactly one __clip_min, got:\n{block}"
            );
            assert_eq!(
                block.matches("let __clip_max = ").count(),
                1,
                "block {idx} should declare exactly one __clip_max, got:\n{block}"
            );
        }
    }

    #[test]
    fn small_graph_uses_flat_codegen() {
        let graph = build_abs_chain(5);
        let code = format_tokens(graph.codegen());

        // Should have a single Model struct, no Submodule structs
        assert!(code.contains("pub struct Model"));
        assert!(!code.contains("Submodule"));
    }

    #[test]
    fn large_graph_uses_partitioned_codegen() {
        let graph = build_abs_chain(250);
        let code = format_tokens(graph.codegen());

        // Should have Submodule structs and a Model that delegates
        assert!(code.contains("pub struct Submodule1"));
        assert!(code.contains("pub struct Model"));
        assert!(code.contains("submodule1: Submodule1"));

        // Submodules should have their own forward methods
        assert!(code.contains("self.submodule1.forward("));

        // The Model forward should still take `input` and return the final tensor
        assert!(code.contains("pub fn forward(&self, input: Tensor<2>) -> Tensor<2>"));
    }

    #[test]
    fn large_graph_with_partition_disabled_uses_flat_codegen() {
        let graph = build_abs_chain(250);
        let code = format_tokens(graph.with_partition(false).codegen());

        // Should use flat codegen despite exceeding MIN_GRAPH_SIZE
        assert!(code.contains("pub struct Model"));
        assert!(
            !code.contains("Submodule"),
            "partition(false) should prevent submodules"
        );

        // Forward should be directly on Model, not delegated
        assert!(code.contains("pub fn forward(&self, input: Tensor<2>) -> Tensor<2>"));
    }

    #[test]
    fn partitioned_graph_snapshot() {
        // Use a graph just above the threshold (200 nodes) for a manageable snapshot
        let graph = build_abs_chain(200);
        let code = format_tokens(graph.codegen());

        // Verify the overall structure by checking key patterns
        // (Full snapshot would be too long; check structural invariants instead)

        // Must have at least 2 submodules
        assert!(code.contains("Submodule1"));
        assert!(code.contains("Submodule2"));

        // Each submodule must have #[derive(Module, Debug)]
        let module_derive_count = code.matches("#[derive(Module, Debug)]").count();
        // At least 3: one per submodule + one for Model
        assert!(
            module_derive_count >= 3,
            "Expected at least 3 #[derive(Module, Debug)], got {}",
            module_derive_count
        );

        // Model::new should create submodules
        assert!(code.contains("Submodule1::new(device)"));
        assert!(code.contains("Submodule2::new(device)"));

        // No duplicate struct definitions
        let submodule1_count = code.matches("pub struct Submodule1").count();
        assert_eq!(submodule1_count, 1, "Submodule1 defined more than once");
    }

    /// Create a temporary .bpk file for tests that need `with_burnpack`.
    fn temp_bpk() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("burn-onnx-test-{}-{}.bpk", std::process::id(), id));
        std::fs::write(&path, [0u8; 4]).unwrap();
        path
    }

    #[test]
    fn load_strategy_file_generates_from_file_and_from_bytes() {
        let bpk = temp_bpk();
        let graph = build_abs_chain(1).with_burnpack(bpk.clone(), LoadStrategy::File);
        let code = format_tokens(graph.codegen());
        let _ = std::fs::remove_file(bpk);

        assert!(
            code.contains("pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &Device)")
        );
        assert!(code.contains("pub fn from_bytes(bytes: Bytes"));
        assert!(code.contains("impl Default for Model"));
        assert!(code.contains("Self::from_file("));
        assert!(!code.contains("from_embedded"));
        // A load failure must name the offending file and the underlying error;
        // "Failed to load burnpack file" alone sends users hunting for the wrong path.
        assert!(code.contains("Failed to load burnpack file {}: {e}"));
        assert!(code.contains("Failed to load burnpack bytes: {e}"));
        // `from_file` references `std::path::Path`, which is not resolvable from
        // `#![no_std]` consumers unless std is explicitly linked. Pin the opt-in.
        assert!(code.contains("extern crate std;"));
    }

    #[test]
    fn load_strategy_embedded_generates_from_embedded_and_from_bytes() {
        let bpk = temp_bpk();
        let graph = build_abs_chain(1).with_burnpack(bpk.clone(), LoadStrategy::Embedded);
        let code = format_tokens(graph.codegen());
        let _ = std::fs::remove_file(bpk);

        assert!(code.contains("pub fn from_embedded("));
        assert!(code.contains("pub fn from_bytes(bytes: Bytes"));
        assert!(code.contains("impl Default for Model"));
        assert!(code.contains("Self::from_embedded("));
        assert!(code.contains("include_bytes!"));
        assert!(!code.contains("from_file"));
        assert!(!code.contains("extern crate std"));
        // Embedded data has no runtime path, so report the build-time source instead.
        assert!(code.contains("Failed to load embedded burnpack (built from {}): {e}"));
    }

    #[test]
    fn load_strategy_bytes_generates_only_from_bytes() {
        let bpk = temp_bpk();
        let graph = build_abs_chain(1).with_burnpack(bpk.clone(), LoadStrategy::Bytes);
        let code = format_tokens(graph.codegen());
        let _ = std::fs::remove_file(bpk);

        assert!(code.contains("pub fn from_bytes(bytes: Bytes"));
        assert!(!code.contains("from_file"));
        assert!(!code.contains("from_embedded"));
        assert!(!code.contains("impl Default for Model"));
        assert!(!code.contains("extern crate std"));
    }

    #[test]
    fn load_strategy_none_generates_no_loaders() {
        let bpk = temp_bpk();
        let graph = build_abs_chain(1).with_burnpack(bpk.clone(), LoadStrategy::None);
        let code = format_tokens(graph.codegen());
        let _ = std::fs::remove_file(bpk);

        assert!(!code.contains("from_file"));
        assert!(!code.contains("from_bytes"));
        assert!(!code.contains("from_embedded"));
        assert!(!code.contains("impl Default for Model"));
        assert!(!code.contains("extern crate std"));
    }
}
