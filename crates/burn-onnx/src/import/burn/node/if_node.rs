use super::prelude::*;
use super::subgraph_helper;
use std::collections::HashSet;

/// Generate inline code for a subgraph branch (then/else).
///
/// Returns (body_code, output_tuple)
fn generate_branch_code(
    subgraph: &onnx_ir::OnnxGraph,
    outer_scope_inputs: &[Argument],
    scope_ref_names: &[String],
    scope: &mut Scope,
    node_position: usize,
) -> (TokenStream, TokenStream) {
    // For If branches, all scope_ref_names are genuine outer-scope references
    // (no filtering needed, unlike Loop/Scan)
    let exclude_names = HashSet::new();

    // Collect names actually used in this branch to avoid unused variable warnings
    let used_names = subgraph_helper::collect_subgraph_referenced_names(subgraph);

    // Generate outer-scope bindings (only for names actually used in this branch)
    let bindings = subgraph_helper::generate_outer_scope_bindings(
        outer_scope_inputs,
        scope_ref_names,
        &exclude_names,
        Some(&used_names),
        scope,
        node_position,
    );

    // Each branch has its own lexical scope. Sharing clone counts across branches
    // makes identical outer-scope names consume one another's future uses.
    let mut branch_scope = Scope::default();
    let scope = &mut branch_scope;
    subgraph_helper::register_subgraph_scope(subgraph, scope, node_position);

    // Generate forward code
    let forward_code =
        subgraph_helper::generate_subgraph_forward_code(subgraph, scope, node_position);

    // Generate output tuple
    let output_names: Vec<_> = subgraph
        .outputs
        .iter()
        .map(|arg| {
            scope
                .at_position(node_position + subgraph.nodes.len())
                .arg(arg)
        })
        .collect();
    let output_tuple = if output_names.len() == 1 {
        let out = &output_names[0];
        quote! { #out }
    } else {
        quote! { (#(#output_names),*) }
    };

    let body = quote! {
        #bindings
        #forward_code
    };

    (body, output_tuple)
}

/// Find dimensions to unsqueeze so a tensor of `branch_rank` matches `target_rank`.
///
/// Compares static shapes to locate size-1 dimensions in the target that are absent
/// in the branch. Falls back to inserting dims at the end if shapes are unavailable.
fn find_unsqueeze_dims(
    branch_shape: Option<&[Option<usize>]>,
    target_shape: Option<&[Option<usize>]>,
    branch_rank: usize,
    target_rank: usize,
) -> Vec<isize> {
    if let (Some(b_shape), Some(t_shape)) = (branch_shape, target_shape) {
        // Walk both shapes, finding positions in target that are size-1 and
        // missing from the branch shape.
        let mut dims = Vec::new();
        let mut b_idx = 0;
        let mut misaligned = false;
        for (t_idx, t_dim) in t_shape.iter().enumerate() {
            if b_idx < b_shape.len() && b_shape[b_idx] == *t_dim {
                b_idx += 1;
            } else if *t_dim == Some(1) {
                dims.push(t_idx as isize);
            } else {
                misaligned = true;
                log::warn!(
                    "If branch shapes don't align cleanly (branch {:?} vs target {:?}), \
                     falling back to trailing unsqueeze dims",
                    b_shape,
                    t_shape
                );
                break;
            }
        }
        if !misaligned && dims.len() == target_rank - branch_rank {
            return dims;
        }
        if !misaligned {
            log::warn!(
                "If branch shape alignment produced {} dims but expected {} \
                 (branch {:?} vs target {:?}), falling back to trailing unsqueeze dims",
                dims.len(),
                target_rank - branch_rank,
                b_shape,
                t_shape,
            );
        }
    }

    // Fallback: insert trailing dimensions
    (branch_rank as isize..target_rank as isize).collect()
}

impl NodeCodegen for onnx_ir::node::if_node::IfNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // Get condition input (first input)
        let cond_arg = self
            .inputs
            .first()
            .expect("If node requires condition input");

        let cond = match &cond_arg.ty {
            ArgType::ScalarNative(_) => {
                let name = arg_to_ident(cond_arg);
                quote! { #name }
            }
            ArgType::ScalarTensor(dtype) => {
                let cond_tensor = scope.arg(cond_arg);
                on_device_to_native(quote! { #cond_tensor }, dtype)
            }
            ArgType::Tensor(_) => {
                let cond_tensor = scope.arg(cond_arg);
                quote! { #cond_tensor.into_scalar::<bool>() }
            }
            ArgType::Shape(rank) => {
                // Shape comparison result: [i64; N] with 0/1 values.
                let name = arg_to_ident(cond_arg);
                if *rank == 0 {
                    // Empty shape: no elements to test, treat as false.
                    quote! { false }
                } else {
                    quote! { #name[0] != 0 }
                }
            }
        };

        // Get outer-scope reference inputs (all inputs after condition)
        let outer_scope_inputs: Vec<_> = self.inputs.iter().skip(1).cloned().collect();

        // Generate code for then and else branches
        let node_position = scope.node_position();
        let mut then_scope = scope.scope().clone();
        let mut else_scope = scope.scope().clone();
        let (then_body, then_output) = generate_branch_code(
            &self.config.then_branch,
            &outer_scope_inputs,
            &self.config.scope_ref_names,
            &mut then_scope,
            node_position,
        );
        let (else_body, else_output) = generate_branch_code(
            &self.config.else_branch,
            &outer_scope_inputs,
            &self.config.scope_ref_names,
            &mut else_scope,
            node_position,
        );

        // Only one branch executes, so each capture consumes one parent use.
        // A folded condition can leave captures used only by the unreachable branch;
        // consume those explicitly so their producer does not become an unused binding.
        let mut used = subgraph_helper::collect_subgraph_referenced_names(&self.config.then_branch);
        used.extend(subgraph_helper::collect_subgraph_referenced_names(
            &self.config.else_branch,
        ));
        let mut unused_captures = Vec::new();
        for (input, name) in outer_scope_inputs.iter().zip(&self.config.scope_ref_names) {
            let value = scope.arg(input);
            if input.is_dynamic() && !used.contains(name) && !used.contains(&input.name) {
                unused_captures.push(quote! { let _ = #value; });
            }
        }

        // Handle rank mismatches between branches by adding unsqueeze_dims
        // to the lower-rank branch so both arms of the `if` have the same type.
        let then_output = align_branch_output(
            then_output,
            &self.config.then_branch.outputs,
            &self.config.else_branch.outputs,
            &self.outputs,
        );
        let else_output = align_branch_output(
            else_output,
            &self.config.else_branch.outputs,
            &self.config.then_branch.outputs,
            &self.outputs,
        );

        // Generate output variable declarations
        let output_names: Vec<_> = self.outputs.iter().map(arg_to_ident).collect();
        let output_decls = if self.outputs.len() == 1 {
            let out = &output_names[0];
            quote! { let #out }
        } else {
            quote! { let (#(#output_names),*) }
        };

        quote! {
            #(#unused_captures)*
            #output_decls = if #cond {
                #then_body
                #then_output
            } else {
                #else_body
                #else_output
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register imports from subgraph nodes
        let mut register_subgraph_imports = |subgraph: &onnx_ir::OnnxGraph| {
            for node in &subgraph.nodes {
                NodeCodegen::register_imports(node, imports);
            }
        };

        register_subgraph_imports(&self.config.then_branch);
        register_subgraph_imports(&self.config.else_branch);
    }
}

/// Wrap a branch output expression with unsqueeze_dims if its rank is lower
/// than the If node's output rank.
fn align_branch_output(
    branch_output: TokenStream,
    branch_outputs: &[Argument],
    peer_outputs: &[Argument],
    if_outputs: &[Argument],
) -> TokenStream {
    if branch_outputs.len() == 1 && if_outputs.len() == 1 {
        let branch_rank = branch_outputs[0].ty.rank();
        let target_rank = if_outputs[0].ty.rank();

        if branch_rank < target_rank {
            let dims = find_unsqueeze_dims(
                branch_outputs[0].ty.static_shape().map(|v| v.as_slice()),
                peer_outputs[0].ty.static_shape().map(|v| v.as_slice()),
                branch_rank,
                target_rank,
            );
            let target_rank_lit = target_rank;
            return quote! {
                #branch_output.unsqueeze_dims::<#target_rank_lit>(&[#(#dims),*])
            };
        }
    }
    // Multi-output or same-rank: no transformation needed
    branch_output
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::codegen_forward_default;
    use super::*;
    use insta::assert_snapshot;
    use onnx_ir::ir::{BoolStore, DType, TensorType};

    #[test]
    fn repeated_branch_outputs_and_unused_capture() {
        let tensor = |name: &str| {
            Argument::new(
                name,
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
        };
        let mut branch = onnx_ir::OnnxGraph::default();
        branch.outputs = vec![tensor("x"), tensor("x")];
        let node = onnx_ir::node::if_node::IfNode {
            name: "if1".into(),
            inputs: vec![
                Argument::new(
                    "cond",
                    ArgType::ScalarNative(DType::Bool(BoolStore::Native)),
                ),
                tensor("x"),
                tensor("unused"),
            ],
            outputs: vec![tensor("a"), tensor("b")],
            config: onnx_ir::node::if_node::IfConfig {
                then_branch: branch.clone(),
                else_branch: branch,
                scope_ref_names: vec!["x".into(), "unused".into()],
            },
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            cond: bool,
            x: Tensor<1>,
            unused: Tensor<1>,
        ) -> (Tensor<1>, Tensor<1>) {
            let _ = unused;
            let (a, b) = if cond {
                let x = x.clone();
                (x.clone(), x)
            } else {
                let x = x.clone();
                (x.clone(), x)
            };
            (a, b)
        }
        ");
    }
    #[test]
    fn alignment_uses_higher_rank_branch_shape() {
        let arg = |name, dims: Vec<Option<usize>>| {
            Argument::new(
                name,
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: dims.len(),
                    static_shape: Some(dims),
                }),
            )
        };
        let result = align_branch_output(
            quote! { value },
            &[arg("lower", vec![Some(3)])],
            &[arg("higher", vec![Some(1), Some(3)])],
            &[arg("out", vec![None, None])],
        );
        assert_snapshot!(result.to_string(), @"value . unsqueeze_dims :: < 2usize > (& [0isize])");
    }
}
