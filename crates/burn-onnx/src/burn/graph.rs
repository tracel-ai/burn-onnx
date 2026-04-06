use super::{BurnImports, Scope, ToTokens};
use crate::LoadStrategy;
use crate::burn::node::NodeCodegen;
use crate::burn::partition::{
    MIN_GRAPH_SIZE, Partition, reorder_constants_to_consumers, try_partition,
};
use burn_store::{BurnpackWriter, TensorSnapshot};
use onnx_ir::{Node, ir::ArgType};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::{collections::HashMap, path::PathBuf};

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
    /// - Outputs: `.into_scalar().elem::<T>()` before the return
    /// - Inputs: `Tensor::from_data([name as T], &self.device)` after the params
    boundary_output_conversions: HashMap<String, onnx_ir::ir::DType>,
    boundary_input_conversions: HashMap<String, onnx_ir::ir::DType>,
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
        // Collect all tensor snapshots from nodes
        let snapshots = self.collect_all_snapshots();

        // Write burnpack file
        let burnpack_file = out_file.with_extension("bpk");
        BurnpackWriter::new(snapshots)
            .with_metadata("producer", "burn-onnx")
            .write_to_file(&burnpack_file)
            .expect("Failed to write burnpack file");

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
        collect_snapshots_from_nodes(&self.nodes, "", &mut field_name_counts, &mut snapshots);
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

    /// Generate tokens representing the graph with Burn modules and tensor operations.
    pub fn codegen(mut self) -> TokenStream {
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

            impl<B: Backend> Model<B> {
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
            // forward converts them to Tensor<B, 1> before calling submodule.forward().
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
            let chunk_fields = collect_fields_for_nodes(chunk_nodes);

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
                let code = NodeCodegen::forward(node, &mut scope_at_pos);
                forward_body.extend(code);
            }

            let submodule_def = quote! {
                #[derive(Module, Debug)]
                pub struct #struct_name<B: Backend> {
                    #(#struct_fields)*
                    phantom: core::marker::PhantomData<B>,
                    #[module(skip)]
                    device: B::Device,
                }

                impl<B: Backend> #struct_name<B> {
                    #[allow(unused_variables)]
                    pub fn new(device: &B::Device) -> Self {
                        #field_init_code
                        Self {
                            #(#field_names_for_init,)*
                            phantom: core::marker::PhantomData,
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
            submodule_field_decls.push(quote! { #field_name: #struct_name<B>, });
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
            pub struct Model<B: Backend> {
                #(#submodule_field_decls)*
                phantom: core::marker::PhantomData<B>,
                #[module(skip)]
                device: B::Device,
            }
            #maybe_blank

            #codegen_default

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    #(#submodule_field_inits)*
                    Self {
                        #(#submodule_field_names,)*
                        phantom: core::marker::PhantomData,
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
            .for_each(|node| NodeCodegen::register_imports(node, &mut self.imports));
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
                let file = file.to_str().unwrap();
                default_impl = quote! {
                    impl<B: Backend> Default for Model<B> {
                        fn default() -> Self {
                            Self::from_file(#file, &Default::default())
                        }
                    }
                    _blank_!();
                };
                extra_loaders = quote! {
                    /// Load model weights from a burnpack file.
                    pub fn from_file(file: &str, device: &B::Device) -> Self {
                        let mut model = Self::new(device);
                        let mut store = BurnpackStore::from_file(file);
                        model.load_from(&mut store).expect("Failed to load burnpack file");
                        model
                    }
                    _blank_!();
                };
            }
            LoadStrategy::Embedded => {
                let file_size = std::fs::metadata(&file)
                    .expect("Failed to read burnpack file metadata")
                    .len() as usize;
                let file = file.to_str().unwrap();
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
                    impl<B: Backend> Default for Model<B> {
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
                    /// Note: Some backends (e.g., NdArray) may still copy data internally.
                    /// See <https://github.com/tracel-ai/burn/issues/4153> for true backend zero-copy.
                    ///
                    /// See <https://github.com/tracel-ai/burn/issues/4123>
                    pub fn from_embedded(device: &B::Device) -> Self {
                        let mut model = Self::new(device);
                        let mut store = BurnpackStore::from_static(EMBEDDED_STATES);
                        model.load_from(&mut store).expect("Failed to load embedded burnpack");
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
            impl<B: Backend> Model<B> {
                #extra_loaders
                /// Load model weights from in-memory bytes.
                ///
                /// The bytes must be the contents of a `.bpk` file.
                pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
                    let mut model = Self::new(device);
                    let mut store = BurnpackStore::from_bytes(Some(bytes));
                    model.load_from(&mut store).expect("Failed to load burnpack bytes");
                    model
                }
            }
        });
    }

    /// Recursively collect all fields from nodes, including subgraph nodes in If/Loop/Scan
    fn collect_all_fields(&self) -> Vec<FieldTuple> {
        collect_fields_for_nodes(&self.nodes)
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

        // Extend with phantom data to avoid unused generic type.
        body.extend(quote! {
            phantom: core::marker::PhantomData<B>,
            #[module(skip)]
            device: B::Device,
        });

        quote! {
            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
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
            pub fn new(device: &B::Device) -> Self {
                #body

                Self {
                    #(#field_names,)*
                    phantom: core::marker::PhantomData,
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
            let code = NodeCodegen::forward(node, &mut scope_at_pos);
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
        // forward() signatures use native types (f32, i64, etc.) not Tensor<B, 1>
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
                        let #name = Tensor::<B, 1>::from_data(
                            burn::tensor::TensorData::from([#name]),
                            (&self.device, #dtype_tokens)
                        );
                    });
                } else if dtype.is_int() || dtype.is_uint() {
                    tokens.extend(quote! {
                        let #name = Tensor::<B, 1, Int>::from_data(
                            burn::tensor::TensorData::from([#name]),
                            (&self.device, #dtype_tokens)
                        );
                    });
                } else if dtype.is_bool() {
                    tokens.extend(quote! {
                        let #name = Tensor::<B, 1, Bool>::from_data(
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

/// Collect fields from a slice of nodes (including If/Loop subgraph fields).
fn collect_fields_for_nodes(nodes: &[Node]) -> Vec<FieldTuple> {
    let mut field_name_counts: HashMap<String, usize> = HashMap::new();
    let mut all_fields: Vec<FieldTuple> = Vec::new();

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
        if let Some(field) = NodeCodegen::field(node) {
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
) {
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

            let node = AbsNodeBuilder::new(&format!("abs{}", i))
                .input_tensor(&in_name, 2, DType::F32)
                .output_tensor(&out_name, 2, DType::F32)
                .build();

            graph.register(Node::Abs(node));
        }

        let last_out = format!("t{}", n - 1);
        graph.register_input_output(vec!["input".to_string()], vec![last_out], &[], &[]);

        graph
    }

    #[test]
    fn small_graph_uses_flat_codegen() {
        let graph = build_abs_chain(5);
        let code = format_tokens(graph.codegen());

        // Should have a single Model struct, no Submodule structs
        assert!(code.contains("pub struct Model<B: Backend>"));
        assert!(!code.contains("Submodule"));
    }

    #[test]
    fn large_graph_uses_partitioned_codegen() {
        let graph = build_abs_chain(250);
        let code = format_tokens(graph.codegen());

        // Should have Submodule structs and a Model that delegates
        assert!(code.contains("pub struct Submodule1<B: Backend>"));
        assert!(code.contains("pub struct Model<B: Backend>"));
        assert!(code.contains("submodule1: Submodule1<B>"));

        // Submodules should have their own forward methods
        assert!(code.contains("self.submodule1.forward("));

        // The Model forward should still take `input` and return the final tensor
        assert!(code.contains("pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>"));
    }

    #[test]
    fn large_graph_with_partition_disabled_uses_flat_codegen() {
        let graph = build_abs_chain(250);
        let code = format_tokens(graph.with_partition(false).codegen());

        // Should use flat codegen despite exceeding MIN_GRAPH_SIZE
        assert!(code.contains("pub struct Model<B: Backend>"));
        assert!(
            !code.contains("Submodule"),
            "partition(false) should prevent submodules"
        );

        // Forward should be directly on Model, not delegated
        assert!(code.contains("pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>"));
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

        assert!(code.contains("pub fn from_file("));
        assert!(code.contains("pub fn from_bytes(bytes: Bytes"));
        assert!(code.contains("impl<B: Backend> Default for Model<B>"));
        assert!(code.contains("Self::from_file("));
        assert!(!code.contains("from_embedded"));
    }

    #[test]
    fn load_strategy_embedded_generates_from_embedded_and_from_bytes() {
        let bpk = temp_bpk();
        let graph = build_abs_chain(1).with_burnpack(bpk.clone(), LoadStrategy::Embedded);
        let code = format_tokens(graph.codegen());
        let _ = std::fs::remove_file(bpk);

        assert!(code.contains("pub fn from_embedded("));
        assert!(code.contains("pub fn from_bytes(bytes: Bytes"));
        assert!(code.contains("impl<B: Backend> Default for Model<B>"));
        assert!(code.contains("Self::from_embedded("));
        assert!(code.contains("include_bytes!"));
        assert!(!code.contains("from_file"));
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
        assert!(!code.contains("impl<B: Backend> Default for Model<B>"));
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
        assert!(!code.contains("impl<B: Backend> Default for Model<B>"));
    }
}
