use super::prelude::*;
use crate::burn::argument_helpers::{elem_cast_tokens, tensor_type_tokens};
use onnx_ir::node::einsum::ParsedEinsum;

/// Compute a product of dimension expressions, or `1usize` when empty.
fn dims_product(dims: &[proc_macro2::TokenStream]) -> proc_macro2::TokenStream {
    if dims.is_empty() {
        quote! { 1usize }
    } else {
        let mut result = dims[0].clone();
        for d in &dims[1..] {
            result = quote! { #result * #d };
        }
        result
    }
}

fn compile_error_tokens(message: impl Into<String>) -> TokenStream {
    let message = message.into();
    quote! {
        compile_error!(#message);
    }
}

/// Check whether a permutation only swaps the last two elements.
///
/// For example `[0, 1, 3, 2]` returns `true` (leading elements are identity,
/// last two are swapped).  An identity permutation returns `false`.
fn is_last_two_swap(perm: &[usize]) -> bool {
    let n = perm.len();
    n >= 2
        && perm[..n - 2].iter().enumerate().all(|(i, &v)| i == v)
        && perm[n - 2] == n - 1
        && perm[n - 1] == n - 2
}

fn find_axis_positions(order: &[char], labels: &[char]) -> Option<Vec<usize>> {
    order
        .iter()
        .map(|axis| labels.iter().position(|current| current == axis))
        .collect()
}

fn scalar_native_to_tensor(expr: TokenStream, dtype: DType) -> TokenStream {
    let dtype_tokens = dtype.to_tokens();

    // Promote through a wide host literal and let Burn cast to the requested dtype.
    if matches!(dtype, DType::F16 | DType::BF16) {
        quote! {
            Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::from([(#expr).to_f64()]),
                (&self.device, #dtype_tokens)
            )
        }
    } else if matches!(dtype, DType::F32) {
        quote! {
            Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::from([f64::from(#expr)]),
                (&self.device, #dtype_tokens)
            )
        }
    } else if matches!(dtype, DType::F64) {
        quote! {
            Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::from([#expr]),
                (&self.device, #dtype_tokens)
            )
        }
    } else if dtype.is_int() || dtype.is_uint() {
        quote! {
            Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::from([#expr as i64]),
                (&self.device, #dtype_tokens)
            )
        }
    } else if dtype.is_bool() {
        quote! {
            Tensor::<B, 1, burn::tensor::Bool>::from_data(
                burn::tensor::TensorData::from([#expr]),
                (&self.device, #dtype_tokens)
            )
        }
    } else {
        compile_error_tokens(format!("Einsum does not support scalar dtype {:?}", dtype))
    }
}

/// Generate bindings that sum out reduced axes and reshape to the effective rank.
///
/// Returns `(bindings_tokens, var_ident_tokens)` where `var_ident_tokens` refers to the
/// post-reduction tensor variable.
fn sum_reduce_bindings(
    input_expr: &TokenStream,
    var_prefix: &str,
    original_labels: &[char],
    reduced_axes: &[char],
    effective_rank: usize,
    dtype: &DType,
) -> (TokenStream, TokenStream) {
    let pre_var = syn::Ident::new(&format!("{var_prefix}_pre"), proc_macro2::Span::call_site());
    let pre_shape_var = syn::Ident::new(
        &format!("{var_prefix}_pre_s"),
        proc_macro2::Span::call_site(),
    );
    let result_var = syn::Ident::new(&format!("{var_prefix}_r"), proc_macro2::Span::call_site());

    // Positions of reduced axes in the original label order.
    let reduced_positions: Vec<usize> = reduced_axes
        .iter()
        .map(|c| original_labels.iter().position(|l| l == c).unwrap())
        .collect();

    // Positions of kept (non-reduced) axes.
    let kept_positions: Vec<usize> = (0..original_labels.len())
        .filter(|i| !reduced_positions.contains(i))
        .collect();

    // Chain .sum_dim(pos) for each reduced axis (order doesn't matter since dims stay).
    let sum_chain: TokenStream = reduced_positions.iter().fold(
        quote! { #pre_var },
        |acc, &pos| quote! { #acc.sum_dim(#pos) },
    );

    let kept_dims: Vec<proc_macro2::TokenStream> = kept_positions
        .iter()
        .map(|&i| quote! { #pre_shape_var[#i] })
        .collect();

    let result_ty = tensor_type_tokens(effective_rank, dtype);

    let bindings = quote! {
        let #pre_var = #input_expr;
        let #pre_shape_var = #pre_var.dims();
        let #result_var: #result_ty = #sum_chain.reshape([#(#kept_dims),*]);
    };

    (bindings, quote! { #result_var })
}

fn einsum_operand_expr(
    node_name: &str,
    equation: &str,
    side: &str,
    labels: &[char],
    arg: &Argument,
    expr: TokenStream,
) -> Result<TokenStream, TokenStream> {
    match &arg.ty {
        ArgType::Tensor(_) => Ok(expr),
        ArgType::ScalarTensor(_) if labels.is_empty() => Ok(expr),
        ArgType::ScalarNative(dtype) if labels.is_empty() => {
            Ok(scalar_native_to_tensor(expr, *dtype))
        }
        ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => Err(compile_error_tokens(format!(
            "Einsum node '{}' uses scalar {} input for non-empty term '{}' in '{}'",
            node_name,
            side,
            labels.iter().collect::<String>(),
            equation
        ))),
        _ => Err(compile_error_tokens(format!(
            "Einsum node '{}' requires tensor-compatible {} input, got {:?}",
            node_name, side, arg.ty
        ))),
    }
}

impl NodeCodegen for onnx_ir::node::einsum::EinsumNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let [lhs_arg, rhs_arg] = self.inputs.as_slice() else {
            return compile_error_tokens(format!(
                "Einsum node '{}' expects exactly 2 inputs, got {}",
                self.name,
                self.inputs.len()
            ));
        };
        let [output_arg] = self.outputs.as_slice() else {
            return compile_error_tokens(format!(
                "Einsum node '{}' expects exactly 1 output, got {}",
                self.name,
                self.outputs.len()
            ));
        };

        let lhs = scope.arg(lhs_arg);
        let rhs = scope.arg(rhs_arg);
        let output = arg_to_ident(output_arg);

        let parsed = match ParsedEinsum::parse(&self.config.equation) {
            Ok(parsed) => parsed,
            Err(error) => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' has an invalid equation '{}': {}",
                    self.name, self.config.equation, error
                ));
            }
        };

        let output_rank = parsed.output.len();
        let result_rank = if output_rank == 0 { 1 } else { output_rank };

        let lhs = match einsum_operand_expr(
            &self.name,
            &self.config.equation,
            "lhs",
            &parsed.lhs,
            lhs_arg,
            lhs,
        ) {
            Ok(lhs) => lhs,
            Err(error) => return error,
        };
        let rhs = match einsum_operand_expr(
            &self.name,
            &self.config.equation,
            "rhs",
            &parsed.rhs,
            rhs_arg,
            rhs,
        ) {
            Ok(rhs) => rhs,
            Err(error) => return error,
        };

        let batch = parsed.batch_axes();
        let contract = parsed.contraction_axes();
        let free_lhs = parsed.free_lhs_axes();
        let free_rhs = parsed.free_rhs_axes();
        let reduced_lhs = parsed.reduced_lhs_axes();
        let reduced_rhs = parsed.reduced_rhs_axes();
        let has_reductions = !reduced_lhs.is_empty() || !reduced_rhs.is_empty();

        // Effective labels after summing out reduced axes.
        let effective_lhs: Vec<char> = parsed
            .lhs
            .iter()
            .filter(|c| !reduced_lhs.contains(c))
            .copied()
            .collect();
        let effective_rhs: Vec<char> = parsed
            .rhs
            .iter()
            .filter(|c| !reduced_rhs.contains(c))
            .copied()
            .collect();

        let lhs_perm_order: Vec<char> = batch
            .iter()
            .chain(free_lhs.iter())
            .chain(contract.iter())
            .copied()
            .collect();
        let lhs_perm = match find_axis_positions(&lhs_perm_order, &effective_lhs) {
            Some(perm) => perm,
            None => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' references an unknown lhs axis in '{}'",
                    self.name, self.config.equation
                ));
            }
        };

        let rhs_perm_order: Vec<char> = batch
            .iter()
            .chain(contract.iter())
            .chain(free_rhs.iter())
            .copied()
            .collect();
        let rhs_perm = match find_axis_positions(&rhs_perm_order, &effective_rhs) {
            Some(perm) => perm,
            None => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' references an unknown rhs axis in '{}'",
                    self.name, self.config.equation
                ));
            }
        };

        let matmul_output_order: Vec<char> = batch
            .iter()
            .chain(free_lhs.iter())
            .chain(free_rhs.iter())
            .copied()
            .collect();
        let output_perm = match find_axis_positions(&parsed.output, &matmul_output_order) {
            Some(perm) => perm,
            None => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' references an unknown output axis in '{}'",
                    self.name, self.config.equation
                ));
            }
        };

        let n_batch = batch.len();
        let n_free_lhs = free_lhs.len();
        let n_contract = contract.len();
        let n_free_rhs = free_rhs.len();

        let lhs_perm_rank = effective_lhs.len();
        let rhs_perm_rank = effective_rhs.len();

        let lhs_is_identity = lhs_perm.iter().enumerate().all(|(i, &v)| i == v);
        let rhs_is_identity = rhs_perm.iter().enumerate().all(|(i, &v)| i == v);
        let output_is_identity = output_perm.iter().enumerate().all(|(i, &v)| i == v);

        // Fast path: scalar broadcast (e.g. ",ij->ij" or "ij,->ij").
        // A scalar times a tensor is just element-wise multiply, no matmul needed.
        // The scalar operand is reshaped to all-1s at the tensor's rank so Burn
        // can broadcast the multiply.
        if !has_reductions && output_is_identity && parsed.lhs.is_empty() && !parsed.rhs.is_empty()
        {
            let ones: Vec<proc_macro2::TokenStream> =
                (0..parsed.rhs.len()).map(|_| quote! { 1usize }).collect();
            return quote! {
                let #output = #rhs.mul(#lhs.reshape([#(#ones),*]));
            };
        }
        if !has_reductions && output_is_identity && parsed.rhs.is_empty() && !parsed.lhs.is_empty()
        {
            let ones: Vec<proc_macro2::TokenStream> =
                (0..parsed.lhs.len()).map(|_| quote! { 1usize }).collect();
            return quote! {
                let #output = #lhs.mul(#rhs.reshape([#(#ones),*]));
            };
        }

        // Fast path: direct N-D matmul (any number of batch dims).
        // Burn's matmul contracts the last two dims and broadcasts leading dims,
        // so when each axis group has exactly 1 element and the axes are already
        // in canonical order [batch..., M, K] x [batch..., K, N] we can call
        // matmul directly, avoiding the reshape-to-3D round-trip.
        let is_direct_matmul = !has_reductions
            && output_is_identity
            && n_free_lhs == 1
            && n_contract == 1
            && n_free_rhs == 1;

        if is_direct_matmul && lhs_is_identity && rhs_is_identity {
            return quote! {
                let #output = #lhs.matmul(#rhs);
            };
        }

        // Fast path: matmul with swap_dims on one or both operands.
        // When the only needed permutation is swapping the last two axes of an
        // operand (e.g. "ij,kj->ik" needs rhs transposed), swap_dims is a
        // zero-copy view, much cheaper than permute + reshape + matmul + reshape.
        if is_direct_matmul {
            let lhs_swap = is_last_two_swap(&lhs_perm);
            let rhs_swap = is_last_two_swap(&rhs_perm);

            if (lhs_is_identity || lhs_swap) && (rhs_is_identity || rhs_swap) {
                let lhs_rank = lhs_perm_rank;
                let rhs_rank = rhs_perm_rank;
                let lhs_expr = if lhs_swap {
                    let d0 = lhs_rank - 2;
                    let d1 = lhs_rank - 1;
                    quote! { #lhs.swap_dims(#d0, #d1) }
                } else {
                    lhs.clone()
                };
                let rhs_expr = if rhs_swap {
                    let d0 = rhs_rank - 2;
                    let d1 = rhs_rank - 1;
                    quote! { #rhs.swap_dims(#d0, #d1) }
                } else {
                    rhs.clone()
                };
                return quote! {
                    let #output = #lhs_expr.matmul(#rhs_expr);
                };
            }
        }

        // Generate pre-reduction code for inputs with one-sided reduced axes.
        // Each reduced axis is summed (producing size 1), then reshaped away.
        let (lhs_reduction_bindings, lhs_input_expr) = if reduced_lhs.is_empty() {
            (quote! {}, lhs.clone())
        } else {
            sum_reduce_bindings(
                &lhs,
                "einsum_lhs",
                &parsed.lhs,
                &reduced_lhs,
                lhs_perm_rank,
                &lhs_arg.ty.elem_type(),
            )
        };
        let (rhs_reduction_bindings, rhs_input_expr) = if reduced_rhs.is_empty() {
            (quote! {}, rhs.clone())
        } else {
            sum_reduce_bindings(
                &rhs,
                "einsum_rhs",
                &parsed.rhs,
                &reduced_rhs,
                rhs_perm_rank,
                &rhs_arg.ty.elem_type(),
            )
        };

        let lhs_perm_expr = if lhs_is_identity {
            lhs_input_expr
        } else {
            let perm_dims: Vec<proc_macro2::TokenStream> =
                lhs_perm.iter().map(|&d| quote! { #d }).collect();
            quote! { #lhs_input_expr.permute([#(#perm_dims),*]) }
        };

        let rhs_perm_expr = if rhs_is_identity {
            rhs_input_expr
        } else {
            let perm_dims: Vec<proc_macro2::TokenStream> =
                rhs_perm.iter().map(|&d| quote! { #d }).collect();
            quote! { #rhs_input_expr.permute([#(#perm_dims),*]) }
        };

        // Flatten grouped axes to a batched `[B, M, K] x [B, K, N]` matmul.
        let lhs_batch_dims: Vec<proc_macro2::TokenStream> = (0..n_batch)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();
        let lhs_free_dims: Vec<proc_macro2::TokenStream> = (n_batch..n_batch + n_free_lhs)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();
        let lhs_contract_dims: Vec<proc_macro2::TokenStream> = (n_batch + n_free_lhs
            ..lhs_perm_rank)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();

        let rhs_free_dims: Vec<proc_macro2::TokenStream> = (n_batch + n_contract..rhs_perm_rank)
            .map(|i| quote! { einsum_rhs_shape[#i] })
            .collect();

        let batch_product = dims_product(&lhs_batch_dims);
        let m_product = dims_product(&lhs_free_dims);
        let k_product = dims_product(&lhs_contract_dims);
        let n_product = dims_product(&rhs_free_dims);

        let batch_shape_dims: Vec<proc_macro2::TokenStream> = (0..n_batch)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();
        let free_lhs_shape_dims: Vec<proc_macro2::TokenStream> = (n_batch..n_batch + n_free_lhs)
            .map(|i| quote! { einsum_lhs_shape[#i] })
            .collect();
        let free_rhs_shape_dims: Vec<proc_macro2::TokenStream> = (n_batch + n_contract
            ..rhs_perm_rank)
            .map(|i| quote! { einsum_rhs_shape[#i] })
            .collect();

        let out_shape_dims: Vec<proc_macro2::TokenStream> = batch_shape_dims
            .iter()
            .chain(free_lhs_shape_dims.iter())
            .chain(free_rhs_shape_dims.iter())
            .cloned()
            .collect();
        let needs_lhs_shape = !lhs_batch_dims.is_empty()
            || !lhs_free_dims.is_empty()
            || !lhs_contract_dims.is_empty()
            || !batch_shape_dims.is_empty()
            || !free_lhs_shape_dims.is_empty();
        let needs_rhs_shape = !rhs_free_dims.is_empty() || !free_rhs_shape_dims.is_empty();

        let lhs_3d_ty = tensor_type_tokens(3, &lhs_arg.ty.elem_type());
        let rhs_3d_ty = tensor_type_tokens(3, &rhs_arg.ty.elem_type());
        let result_ty = tensor_type_tokens(result_rank, &output_arg.ty.elem_type());

        let result_expr = if output_is_identity {
            quote! { einsum_result }
        } else {
            let out_perm_dims: Vec<proc_macro2::TokenStream> =
                output_perm.iter().map(|&d| quote! { #d }).collect();
            quote! { einsum_result.permute([#(#out_perm_dims),*]) }
        };
        let final_output_expr = match &output_arg.ty {
            ArgType::Tensor(tensor) => {
                if tensor.rank != output_rank {
                    return compile_error_tokens(format!(
                        "Einsum node '{}' expected tensor output rank {} but got {}",
                        self.name, output_rank, tensor.rank
                    ));
                }
                result_expr
            }
            ArgType::ScalarTensor(_) => {
                if output_rank != 0 {
                    return compile_error_tokens(format!(
                        "Einsum node '{}' expected scalar tensor output for rank-0 equation '{}'",
                        self.name, self.config.equation
                    ));
                }
                result_expr
            }
            ArgType::ScalarNative(dtype) => {
                if output_rank != 0 {
                    return compile_error_tokens(format!(
                        "Einsum node '{}' expected scalar output for rank-0 equation '{}'",
                        self.name, self.config.equation
                    ));
                }
                let cast = elem_cast_tokens(dtype);
                quote! { (#result_expr).into_scalar()#cast }
            }
            _ => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' does not support output type {:?}",
                    self.name, output_arg.ty
                ));
            }
        };
        let lhs_shape_binding = if needs_lhs_shape {
            quote! { let einsum_lhs_shape = einsum_lhs.dims(); }
        } else {
            quote! {}
        };
        let rhs_shape_binding = if needs_rhs_shape {
            quote! { let einsum_rhs_shape = einsum_rhs.dims(); }
        } else {
            quote! {}
        };
        let result_reshape_expr = if output_rank == 0 {
            quote! { .reshape([1usize]) }
        } else {
            quote! { .reshape([#(#out_shape_dims),*]) }
        };

        quote! {
            let #output = {
                #lhs_reduction_bindings
                #rhs_reduction_bindings
                let einsum_lhs = #lhs_perm_expr;
                let einsum_rhs = #rhs_perm_expr;
                #lhs_shape_binding
                #rhs_shape_binding
                let einsum_lhs_3d: #lhs_3d_ty =
                    einsum_lhs.reshape([#batch_product, #m_product, #k_product]);
                let einsum_rhs_3d: #rhs_3d_ty =
                    einsum_rhs.reshape([#batch_product, #k_product, #n_product]);
                let einsum_result: #result_ty = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    #result_reshape_expr;
                #final_output_expr
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::einsum::{EinsumConfig, EinsumNodeBuilder};

    #[test]
    fn test_einsum_simple_matmul() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "ij,jk->ik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.matmul(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_einsum_batch_matmul() {
        let node = EinsumNodeBuilder::new("einsum2")
            .input_tensor("lhs", 3, DType::F32)
            .input_tensor("rhs", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(EinsumConfig {
                equation: "bij,bjk->bik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = lhs.matmul(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_einsum_simple_matmul_int() {
        let node = EinsumNodeBuilder::new("einsum_int")
            .input_tensor("lhs", 2, DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(EinsumConfig {
                equation: "ij,jk->ik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            lhs: Tensor<B, 2, Int>,
            rhs: Tensor<B, 2, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = lhs.matmul(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_einsum_sam_pattern() {
        let node = EinsumNodeBuilder::new("einsum3")
            .input_tensor("r_q", 4, DType::F32)
            .input_tensor("r_h", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(EinsumConfig {
                equation: "bhwc,hkc->bhwk".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, r_q: Tensor<B, 4>, r_h: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let einsum_lhs = r_q.permute([1usize, 0usize, 2usize, 3usize]);
                let einsum_rhs = r_h.permute([0usize, 2usize, 1usize]);
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[1usize] * einsum_lhs_shape[2usize],
                        einsum_lhs_shape[3usize],
                    ]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[3usize],
                        einsum_rhs_shape[2usize],
                    ]);
                let einsum_result: Tensor<B, 4> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[1usize],
                        einsum_lhs_shape[2usize],
                        einsum_rhs_shape[2usize],
                    ]);
                einsum_result.permute([1usize, 0usize, 2usize, 3usize])
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_outer_product() {
        let node = EinsumNodeBuilder::new("einsum4")
            .input_tensor("a", 1, DType::F32)
            .input_tensor("b", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "i,j->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = {
                let einsum_lhs = a;
                let einsum_rhs = b;
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([1usize, einsum_lhs_shape[0usize], 1usize]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([1usize, 1usize, einsum_rhs_shape[0usize]]);
                let einsum_result: Tensor<B, 2> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([einsum_lhs_shape[0usize], einsum_rhs_shape[0usize]]);
                einsum_result
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_outer_product_int_preserves_tensor_kind() {
        let node = EinsumNodeBuilder::new("einsum_outer_int")
            .input_tensor("a", 1, DType::I32)
            .input_tensor("b", 1, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(EinsumConfig {
                equation: "i,j->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, a: Tensor<B, 1, Int>, b: Tensor<B, 1, Int>) -> Tensor<B, 2, Int> {
            let output = {
                let einsum_lhs = a;
                let einsum_rhs = b;
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3, Int> = einsum_lhs
                    .reshape([1usize, einsum_lhs_shape[0usize], 1usize]);
                let einsum_rhs_3d: Tensor<B, 3, Int> = einsum_rhs
                    .reshape([1usize, 1usize, einsum_rhs_shape[0usize]]);
                let einsum_result: Tensor<B, 2, Int> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([einsum_lhs_shape[0usize], einsum_rhs_shape[0usize]]);
                einsum_result
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_accepts_scalar_native_lhs() {
        let node = EinsumNodeBuilder::new("einsum_scalar_lhs")
            .input_scalar("scale", DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: ",ij->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains(".mul("));
        assert!(code.contains("from_data("));
        assert!(code.contains("f64::from(scale)"));
        assert!(code.contains("reshape([1usize, 1usize])"));
        assert!(!code.contains("matmul"));
    }

    #[test]
    fn test_einsum_accepts_float16_scalar_native_lhs() {
        let node = EinsumNodeBuilder::new("einsum_scalar_lhs_f16")
            .input_scalar("scale", DType::F16)
            .input_tensor("rhs", 2, DType::F16)
            .output_tensor("output", 2, DType::F16)
            .config(EinsumConfig {
                equation: ",ij->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("rhs"));
        assert!(code.contains(".mul("));
        assert!(code.contains("(scale).to_f64()"));
        assert!(!code.contains("matmul"));
    }

    #[test]
    fn test_einsum_accepts_scalar_tensor_rhs() {
        let node = EinsumNodeBuilder::new("einsum_scalar_rhs")
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar_tensor("scale", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "ij,->ij".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("lhs.mul(scale.reshape([1usize, 1usize]))"));
        assert!(!code.contains("matmul"));
    }

    #[test]
    fn test_einsum_scalar_scalar_can_return_scalar_tensor_output() {
        let node = EinsumNodeBuilder::new("einsum_scalar_scalar")
            .input_scalar("lhs", DType::F32)
            .input_scalar_tensor("rhs", DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .config(EinsumConfig {
                equation: ",->".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);

        assert!(code.contains("-> Tensor<B, 1>"));
        assert!(code.contains("let einsum_lhs ="));
        assert!(code.contains("from_data("));
        assert!(code.contains("let einsum_rhs = rhs;"));
        assert!(code.contains("let einsum_result: Tensor<B, 1>"));
        assert!(code.contains(".reshape([1usize])"));
    }

    #[test]
    fn test_einsum_scalar_scalar_native_output() {
        let node = EinsumNodeBuilder::new("einsum_scalar_scalar_native")
            .input_scalar("lhs", DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_scalar("output", DType::F32)
            .config(EinsumConfig {
                equation: ",->".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);

        assert!(code.contains("-> f32"));
        assert!(code.contains("TensorData::from([f64::from(lhs)])"));
        assert!(code.contains("TensorData::from([f64::from(rhs)])"));
        assert!(code.contains("let einsum_result: Tensor<B, 1>"));
        assert!(code.contains(".reshape([1usize])"));
        assert!(code.contains("(einsum_result).into_scalar().elem::<f32>()"));
    }

    #[test]
    fn test_einsum_swap_dims_matmul() {
        let node = EinsumNodeBuilder::new("einsum_scope")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("einsum_rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "ij,kj->ik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, einsum_rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.matmul(einsum_rhs.swap_dims(0usize, 1usize));
            output
        }
        ");
    }

    #[test]
    fn test_einsum_batched_swap_dims_matmul() {
        let node = EinsumNodeBuilder::new("einsum_batched_swap")
            .input_tensor("q", 3, DType::F32)
            .input_tensor("k", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(EinsumConfig {
                equation: "bij,bkj->bik".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = q.matmul(k.swap_dims(1usize, 2usize));
            output
        }
        ");
    }

    #[test]
    fn test_einsum_sam_pattern_2() {
        let node = EinsumNodeBuilder::new("einsum5")
            .input_tensor("r_q", 4, DType::F32)
            .input_tensor("r_w", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(EinsumConfig {
                equation: "bhwc,wkc->bhwk".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, r_q: Tensor<B, 4>, r_w: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let einsum_lhs = r_q.permute([2usize, 0usize, 1usize, 3usize]);
                let einsum_rhs = r_w.permute([0usize, 2usize, 1usize]);
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[1usize] * einsum_lhs_shape[2usize],
                        einsum_lhs_shape[3usize],
                    ]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[3usize],
                        einsum_rhs_shape[2usize],
                    ]);
                let einsum_result: Tensor<B, 4> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([
                        einsum_lhs_shape[0usize],
                        einsum_lhs_shape[1usize],
                        einsum_lhs_shape[2usize],
                        einsum_rhs_shape[2usize],
                    ]);
                einsum_result.permute([1usize, 2usize, 0usize, 3usize])
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_one_sided_reduction() {
        let node = EinsumNodeBuilder::new("einsum_reduce")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "ij,kl->il".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let einsum_lhs_pre = lhs;
                let einsum_lhs_pre_s = einsum_lhs_pre.dims();
                let einsum_lhs_r: Tensor<B, 1> = einsum_lhs_pre
                    .sum_dim(1usize)
                    .reshape([einsum_lhs_pre_s[0usize]]);
                let einsum_rhs_pre = rhs;
                let einsum_rhs_pre_s = einsum_rhs_pre.dims();
                let einsum_rhs_r: Tensor<B, 1> = einsum_rhs_pre
                    .sum_dim(0usize)
                    .reshape([einsum_rhs_pre_s[1usize]]);
                let einsum_lhs = einsum_lhs_r;
                let einsum_rhs = einsum_rhs_r;
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([1usize, einsum_lhs_shape[0usize], 1usize]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([1usize, 1usize, einsum_rhs_shape[0usize]]);
                let einsum_result: Tensor<B, 2> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([einsum_lhs_shape[0usize], einsum_rhs_shape[0usize]]);
                einsum_result
            };
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_one_sided_reduction_lhs_only() {
        let node = EinsumNodeBuilder::new("einsum_reduce_lhs")
            .input_tensor("a", 3, DType::F32)
            .input_tensor("b", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(EinsumConfig {
                equation: "ijk,l->il".to_string(),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = {
                let einsum_lhs_pre = a;
                let einsum_lhs_pre_s = einsum_lhs_pre.dims();
                let einsum_lhs_r: Tensor<B, 1> = einsum_lhs_pre
                    .sum_dim(1usize)
                    .sum_dim(2usize)
                    .reshape([einsum_lhs_pre_s[0usize]]);
                let einsum_lhs = einsum_lhs_r;
                let einsum_rhs = b;
                let einsum_lhs_shape = einsum_lhs.dims();
                let einsum_rhs_shape = einsum_rhs.dims();
                let einsum_lhs_3d: Tensor<B, 3> = einsum_lhs
                    .reshape([1usize, einsum_lhs_shape[0usize], 1usize]);
                let einsum_rhs_3d: Tensor<B, 3> = einsum_rhs
                    .reshape([1usize, 1usize, einsum_rhs_shape[0usize]]);
                let einsum_result: Tensor<B, 2> = einsum_lhs_3d
                    .matmul(einsum_rhs_3d)
                    .reshape([einsum_lhs_shape[0usize], einsum_rhs_shape[0usize]]);
                einsum_result
            };
            output
        }
        "#);
    }
}
