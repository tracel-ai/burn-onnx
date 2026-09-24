use super::prelude::*;
use onnx_ir::node::attention::AttentionQkMatmulOutputMode;

// Semantics follow the current ONNX Attention definition (onnx >= 1.22), which
// changed three things from the onnx 1.19 reference: K/V heads are interleaved for
// grouped-query attention, the causal mask is offset by the cached length, and
// softcap is applied before the mask. The last two agree with burn's attention.

/// Whether this attention node can use `burn::tensor::module::attention()`.
///
/// Only the qk_matmul intermediate output needs the hand-written path.
fn use_burn_attention(node: &onnx_ir::attention::AttentionNode) -> bool {
    node.outputs.get(3).is_none()
}

/// Known size of `axis` of a tensor argument.
fn static_dim(arg: &Argument, axis: usize) -> Option<usize> {
    arg.ty.static_shape()?.get(axis).copied().flatten()
}

/// Setup shared by both paths: `q`, `k` and `v` bound as `[batch, heads, seq, dim]`,
/// the KV cache appended, `past_len` bound when causal masking needs it, and K/V
/// heads repeated for grouped-query attention.
struct Prelude {
    body: TokenStream,
    reshape_output: TokenStream,
    output_names: Vec<Ident>,
    rank: usize,
    /// Whether a KV cache (past_key and past_value) is appended.
    past_kv: bool,
}

fn prelude(node: &onnx_ir::attention::AttentionNode, scope: &mut ScopeAtPosition<'_>) -> Prelude {
    let q = scope.arg(node.inputs.first().unwrap());
    let k = scope.arg(node.inputs.get(1).unwrap());
    let v = scope.arg(node.inputs.get(2).unwrap());
    let output_y = arg_to_ident(node.outputs.first().unwrap());

    let past_kv = matches!((node.inputs.get(4), node.inputs.get(5)), (Some(_), Some(_)));
    let present_kv = matches!(
        (node.outputs.get(1), node.outputs.get(2)),
        (Some(_), Some(_))
    );
    if past_kv != present_kv {
        panic!("Attention: past_[key,value] and present_[key,value] must be used together.")
    }

    let rank = match &node.inputs.first().unwrap().ty {
        ArgType::Tensor(t) => t.rank,
        _ => panic!("Expected tensor input for Q"),
    };

    let mut body = quote! {
        let q = #q;
        let k = #k;
        let v = #v;
    };

    // Reshape rank-3 inputs to rank-4: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    let mut reshape_output = quote! {};
    if rank == 3 {
        let kv_num_heads = node
            .config
            .kv_num_heads
            .expect("kv_num_heads required for rank 3");
        let q_num_heads = node
            .config
            .q_num_heads
            .expect("q_num_heads required for rank 3");

        body.extend(quote! {
            let [batch_size, q_sequence_length, q_hidden_size] = q.dims();
            #[allow(clippy::identity_op)]
            let head_size = q_hidden_size / #q_num_heads;
            let kv_sequence_length = k.dims()[1];
            #[allow(clippy::identity_op)]
            let v_head_size = v.dims()[2] / #kv_num_heads;
            let q = q.reshape([batch_size, q_sequence_length, #q_num_heads, head_size])
                    .permute([0, 2, 1, 3]);
            let k = k.reshape([batch_size, kv_sequence_length, #kv_num_heads, head_size])
                    .permute([0, 2, 1, 3]);
            let v = v.reshape([batch_size, kv_sequence_length, #kv_num_heads, v_head_size])
                    .permute([0, 2, 1, 3]);
        });

        reshape_output = quote! {
            let #output_y = #output_y.permute([0, 2, 1, 3]).reshape([batch_size as i32, q_sequence_length as i32, -1]);
        };
    }

    let mut output_names = vec![output_y];
    if past_kv {
        let past_k = scope.arg(node.inputs.get(4).unwrap());
        let past_v = scope.arg(node.inputs.get(5).unwrap());
        let present_k = arg_to_ident(node.outputs.get(1).unwrap());
        let present_v = arg_to_ident(node.outputs.get(2).unwrap());

        // The causal mask is offset by the length of the cache.
        let new_len = node
            .config
            .is_causal
            .then(|| quote! { let new_len = k.dims()[2]; });
        let past_len = node
            .config
            .is_causal
            .then(|| quote! { let past_len = k.dims()[2] - new_len; });
        body.extend(quote! {
            #new_len
            let #present_k = Tensor::cat([#past_k, k].to_vec(), 2);
            let k = #present_k.clone();
            let #present_v = Tensor::cat([#past_v, v].to_vec(), 2);
            let v = #present_v.clone();
            #past_len
        });
        output_names.push(present_k);
        output_names.push(present_v);
    }
    if let Some(qk_out) = node.outputs.get(3) {
        output_names.push(arg_to_ident(qk_out));
    }

    body.extend(gqa_expand(node, rank));

    Prelude {
        body,
        reshape_output,
        output_names,
        rank,
        past_kv,
    }
}

/// Repeat K and V across heads for grouped-query attention: ONNX shares each K/V
/// head with `q_heads / kv_heads` consecutive query heads (`[h0, h0, h1, h1]`), and
/// burn's attention takes K/V with as many heads as Q. Emitted only when the head
/// counts can differ.
fn gqa_expand(node: &onnx_ir::attention::AttentionNode, rank: usize) -> TokenStream {
    let heads_may_differ = if rank == 3 {
        node.config.q_num_heads != node.config.kv_num_heads
    } else {
        let q_heads = static_dim(&node.inputs[0], 1);
        let kv_heads = static_dim(&node.inputs[1], 1);
        !matches!((q_heads, kv_heads), (Some(q), Some(kv)) if q == kv)
    };
    if !heads_may_differ {
        return quote! {};
    }
    quote! {
        let groups = q.dims()[1] / k.dims()[1];
        let (k, v) = if groups > 1 {
            let [batch, kv_heads, k_len, k_dim] = k.dims();
            let v_dim = v.dims()[3];
            (
                k.unsqueeze_dim::<5>(2)
                    .expand([batch, kv_heads, groups, k_len, k_dim])
                    .reshape([batch, kv_heads * groups, k_len, k_dim]),
                v.unsqueeze_dim::<5>(2)
                    .expand([batch, kv_heads, groups, k_len, v_dim])
                    .reshape([batch, kv_heads * groups, k_len, v_dim]),
            )
        } else {
            (k, v)
        };
    }
}

/// Whether burn's own `is_causal` gives the ONNX mask. burn hides key `j` from
/// query `i` when `j > i + (k_len - q_len)`, ONNX when `j > i + past_len`; with
/// `k_len = past_len + new_len` the two agree exactly when Q and the new K are
/// equally long.
fn native_causal(node: &onnx_ir::attention::AttentionNode, rank: usize) -> bool {
    let seq_axis = if rank == 3 { 1 } else { 2 };
    // With a user mask the causal mask is built explicitly, so the fully hidden row
    // guard sees both: a row can be hidden by the two together.
    let has_user_mask = node.inputs.get(3).is_some_and(|mask| !mask.is_optional());
    node.config.is_causal
        && !has_user_mask
        && matches!(
            (static_dim(&node.inputs[0], seq_axis), static_dim(&node.inputs[1], seq_axis)),
            (Some(q), Some(k)) if q == k
        )
}

/// The ONNX causal mask over the full `[batch, heads, q, k]` score shape: `true`
/// where key `j > i + past_len` is hidden from query `i`. Reads `q`, `k` and, with
/// a cache, `past_len`.
fn causal_mask(past_kv: bool) -> TokenStream {
    let offset = past_kv.then(|| quote! { .add_scalar(past_len as i64) });
    quote! {{
        let [batch, heads, q_len, _] = q.dims();
        let k_len = k.dims()[2];
        let rows = Tensor::<1, Int>::arange(0..q_len as i64, (&self.device, burn::tensor::DType::I64))
            .reshape([q_len, 1])
            #offset
            .expand([q_len, k_len]);
        let cols = Tensor::<1, Int>::arange(0..k_len as i64, (&self.device, burn::tensor::DType::I64))
            .reshape([1, k_len])
            .expand([q_len, k_len]);
        cols.greater(rows)
            .unsqueeze::<4>()
            .expand([batch, heads, q_len, k_len])
    }}
}

/// An ONNX attention mask lifted to rank 4. Masks broadcast right-aligned against
/// `[batch, heads, q, k]`, so lower ranks gain leading axes.
fn lift_mask(mask: TokenStream, rank: usize) -> TokenStream {
    match rank {
        2 | 3 => quote! { #mask.unsqueeze::<4>() },
        4 => mask,
        _ => panic!("Attention mask must be rank 2, 3, or 4"),
    }
}

/// The user mask, as burn's bool mask (`true` hides a key) or as an additive bias.
enum UserMask {
    Hide(TokenStream),
    Bias(TokenStream),
}

fn user_mask(
    node: &onnx_ir::attention::AttentionNode,
    scope: &mut ScopeAtPosition<'_>,
) -> Option<UserMask> {
    let mask_input = node.inputs.get(3).filter(|a| !a.is_optional())?;
    let mask_arg = scope.arg(mask_input);
    let ArgType::Tensor(t) = &mask_input.ty else {
        panic!("Attention mask must be a tensor");
    };
    Some(if t.dtype.is_bool() {
        // ONNX marks positions to attend; burn marks positions to hide.
        UserMask::Hide(lift_mask(quote! { #mask_arg.bool_not() }, t.rank))
    } else if t.dtype.is_float() {
        UserMask::Bias(lift_mask(mask_arg, t.rank))
    } else if t.dtype.is_int() || t.dtype.is_uint() {
        let q_dtype = node.inputs.first().unwrap().ty.elem_type().to_tokens();
        UserMask::Bias(lift_mask(
            quote! { #mask_arg.float().cast(#q_dtype) },
            t.rank,
        ))
    } else {
        panic!("Unsupported attention mask type")
    })
}

impl NodeCodegen for onnx_ir::attention::AttentionNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        if self.config.softmax_precision.is_some() {
            panic!("Attention: non-default softmax precision is not yet supported")
        }

        if use_burn_attention(self) {
            forward_burn_attention(self, scope)
        } else {
            forward_custom(self, scope)
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if !use_burn_attention(self) {
            imports.register("burn::tensor::activation::softmax");
        }
    }
}

/// Generates code using `burn::tensor::module::attention()` which dispatches to
/// optimized backend implementations (e.g. flash attention on GPU).
fn forward_burn_attention(
    node: &onnx_ir::attention::AttentionNode,
    scope: &mut ScopeAtPosition<'_>,
) -> TokenStream {
    let Prelude {
        mut body,
        reshape_output,
        output_names,
        rank,
        past_kv,
    } = prelude(node, scope);
    let output_y = &output_names[0];

    let native_causal = native_causal(node, rank);
    let explicit_causal = node.config.is_causal && !native_causal;

    let scale_tokens = match node.config.scale {
        Some(scale) => quote! { Some(#scale) },
        None => quote! { None },
    };
    let softcap_tokens = if node.config.softcap != 0.0 {
        let softcap = node.config.softcap;
        quote! { Some(#softcap) }
    } else {
        quote! { None }
    };
    let options = quote! {
        burn::tensor::ops::AttentionModuleOptions {
            scale: #scale_tokens,
            softcap: #softcap_tokens,
            is_causal: #native_causal,
        }
    };

    // The masks read Q and K's shapes, so they are bound before Q and K move into
    // the call.
    let (hide, bias) = match user_mask(node, scope) {
        Some(UserMask::Hide(mask)) => (Some(mask), None),
        Some(UserMask::Bias(bias)) => (None, Some(bias)),
        None => (None, None),
    };
    let has_user_mask = hide.is_some() || bias.is_some();
    let mask = match (hide, explicit_causal) {
        (Some(hide), true) => {
            let causal = causal_mask(past_kv);
            body.extend(quote! { let causal = #causal; });
            Some(quote! { #hide.expand(causal.dims()).bool_or(causal) })
        }
        (Some(hide), false) => Some(hide),
        (None, true) => Some(causal_mask(past_kv)),
        (None, false) => None,
    };
    if let Some(mask) = &mask {
        body.extend(quote! { let hide = #mask; });
    }
    if let Some(bias) = &bias {
        body.extend(quote! { let score_bias = #bias; });
    }

    // A query row with every key hidden has no softmax; ONNX defines its output as
    // zeros. The causal mask alone always leaves key 0 visible, so only a user mask
    // can hide a whole row.
    let row_guard = has_user_mask.then(|| {
        let hidden = match (mask.is_some(), bias.is_some()) {
            (true, true) => quote! {{
                let shape = [q.dims()[0], q.dims()[1], q.dims()[2], k.dims()[2]];
                score_bias.clone()
                    .expand(shape)
                    .mask_fill(hide.clone().expand(shape), f32::NEG_INFINITY)
                    .max_dim(3)
                    .equal_elem(f32::NEG_INFINITY)
            }},
            (true, false) => quote! { hide.clone().all_dim(3) },
            _ => quote! { score_bias.clone().max_dim(3).equal_elem(f32::NEG_INFINITY) },
        };
        body.extend(quote! { let hidden_rows = #hidden; });
        quote! {
            let #output_y = {
                let dims = #output_y.dims();
                #output_y.mask_fill(hidden_rows.expand(dims), 0.0)
            };
        }
    });

    let option = |present: bool, name: TokenStream| {
        if present {
            quote! { Some(#name) }
        } else {
            quote! { None }
        }
    };
    let mask_tokens = option(mask.is_some(), quote! { hide });
    let bias_tokens = option(bias.is_some(), quote! { score_bias });

    quote! {
        let (#(#output_names,)*) = {
            #body
            let #output_y = burn::tensor::module::attention(q, k, v, #mask_tokens, #bias_tokens, #options);
            #row_guard
            #reshape_output
            (#(#output_names,)*)
        };
    }
}

/// Hand-written attention for the qk_matmul intermediate output, which burn's
/// attention API does not expose.
fn forward_custom(
    node: &onnx_ir::attention::AttentionNode,
    scope: &mut ScopeAtPosition<'_>,
) -> TokenStream {
    let Prelude {
        mut body,
        reshape_output,
        output_names,
        rank,
        past_kv,
    } = prelude(node, scope);
    let output_y = &output_names[0];

    body.extend(match node.config.scale {
        Some(scale) => {
            let scale = scale.sqrt();
            quote! { let scale = #scale; }
        }
        None if rank == 3 => quote! { let scale = (1.0 / (head_size as f64).sqrt()).sqrt(); },
        None => quote! { let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt(); },
    });

    // Everything that masks a score folds into one additive bias over the full score
    // shape: the user mask (a bool one as 0 / -inf) and the causal mask.
    let user = user_mask(node, scope);
    let masked = user.is_some() || node.config.is_causal;
    if masked {
        let causal = node.config.is_causal.then(|| {
            let causal = causal_mask(past_kv);
            quote! { let score_bias = score_bias.mask_fill(#causal, f32::NEG_INFINITY); }
        });
        let user = match user {
            Some(UserMask::Bias(mask)) => {
                quote! { let score_bias = score_bias + #mask.expand(shape); }
            }
            Some(UserMask::Hide(mask)) => {
                quote! { let score_bias = score_bias.mask_fill(#mask.expand(shape), f32::NEG_INFINITY); }
            }
            None => quote! {},
        };
        body.extend(quote! {
            let score_bias = {
                let shape = [q.dims()[0], q.dims()[1], q.dims()[2], k.dims()[2]];
                let score_bias = Tensor::<4>::zeros(shape, (&self.device, q.dtype()));
                #user
                #causal
                score_bias
            };
        });
    }

    let qk_out = node.outputs.get(3).map(arg_to_ident);
    let capture = |mode: AttentionQkMatmulOutputMode, value: TokenStream| match &qk_out {
        Some(out) if node.config.qk_matmul_output_mode == mode => {
            quote! { let #out = #value.clone(); }
        }
        _ => quote! {},
    };
    let after_matmul = capture(AttentionQkMatmulOutputMode::Matmul, quote! { qk });
    let after_softcap = capture(
        AttentionQkMatmulOutputMode::MatmulAfterSoftcap,
        quote! { qk },
    );
    let after_mask = capture(
        AttentionQkMatmulOutputMode::MatmulPlusAttentionMask,
        quote! { qk },
    );
    let after_softmax = capture(
        AttentionQkMatmulOutputMode::MatmulAfterSoftmax,
        quote! { scores },
    );

    let softcap = (node.config.softcap != 0.0).then(|| {
        let softcap = node.config.softcap;
        let inv_softcap = 1.0 / softcap;
        quote! { let qk = (qk * #inv_softcap).tanh() * #softcap; }
    });
    let (add_bias, row_guard) = if masked {
        (
            quote! { let qk = qk + score_bias.clone(); },
            // A query row with every key hidden has no softmax; ONNX defines it as
            // zeros.
            quote! {
                let scores = {
                    let hidden_rows = score_bias.max_dim(3).equal_elem(f32::NEG_INFINITY);
                    let dims = scores.dims();
                    scores.mask_fill(hidden_rows.expand(dims), 0.0)
                };
            },
        )
    } else {
        (quote! {}, quote! {})
    };

    quote! {
        let (#(#output_names,)*) = {
            #body

            let q_scaled = q * scale;
            let k_scaled = k * scale;
            let k_transpose = k_scaled.transpose();
            let qk = q_scaled.matmul(k_transpose);
            #after_matmul
            #softcap
            #after_softcap
            #add_bias
            #after_mask
            let scores = softmax(qk, 3);
            #row_guard
            #after_softmax
            let #output_y = scores.matmul(v);
            #reshape_output
            (#(#output_names,)*)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::attention::{AttentionConfig, AttentionNodeBuilder, AttentionQkMatmulOutputMode};

    #[test]
    fn test_attention_basic_rank4() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, query: Tensor<4>, key: Tensor<4>, value: Tensor<4>) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_rank3() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: Some(8),
            q_num_heads: Some(8),
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 3, DType::F32)
            .input_tensor("key", 3, DType::F32)
            .input_tensor("value", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, query: Tensor<3>, key: Tensor<3>, value: Tensor<3>) -> Tensor<3> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let [batch_size, q_sequence_length, q_hidden_size] = q.dims();
                #[allow(clippy::identity_op)]
                let head_size = q_hidden_size / 8usize;
                let kv_sequence_length = k.dims()[1];
                #[allow(clippy::identity_op)]
                let v_head_size = v.dims()[2] / 8usize;
                let q = q
                    .reshape([batch_size, q_sequence_length, 8usize, head_size])
                    .permute([0, 2, 1, 3]);
                let k = k
                    .reshape([batch_size, kv_sequence_length, 8usize, head_size])
                    .permute([0, 2, 1, 3]);
                let v = v
                    .reshape([batch_size, kv_sequence_length, 8usize, v_head_size])
                    .permute([0, 2, 1, 3]);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = output
                    .permute([0, 2, 1, 3])
                    .reshape([batch_size as i32, q_sequence_length as i32, -1]);
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_causal_mask() {
        let config = AttentionConfig {
            is_causal: true,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, query: Tensor<4>, key: Tensor<4>, value: Tensor<4>) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let hide = {
                    let [batch, heads, q_len, _] = q.dims();
                    let k_len = k.dims()[2];
                    let rows = Tensor::<
                        1,
                        Int,
                    >::arange(0..q_len as i64, (&self.device, burn::tensor::DType::I64))
                        .reshape([q_len, 1])
                        .expand([q_len, k_len]);
                    let cols = Tensor::<
                        1,
                        Int,
                    >::arange(0..k_len as i64, (&self.device, burn::tensor::DType::I64))
                        .reshape([1, k_len])
                        .expand([q_len, k_len]);
                    cols.greater(rows).unsqueeze::<4>().expand([batch, heads, q_len, k_len])
                };
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(hide),
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_mask() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            mask: Tensor<4>,
        ) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let score_bias = mask;
                let hidden_rows = score_bias.clone().max_dim(3).equal_elem(f32::NEG_INFINITY);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    Some(score_bias),
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_softcap() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 50.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, query: Tensor<4>, key: Tensor<4>, value: Tensor<4>) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: Some(50f64),
                        is_causal: false,
                    },
                );
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_custom_scale() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: Some(0.125),
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, query: Tensor<4>, key: Tensor<4>, value: Tensor<4>) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: Some(0.125f64),
                        softcap: None,
                        is_causal: false,
                    },
                );
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_bool_mask() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 2, DType::Bool(BoolStore::Native))
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            mask: Tensor<2, Bool>,
        ) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let hide = mask.bool_not().unsqueeze::<4>();
                let hidden_rows = hide.clone().all_dim(3);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(hide),
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_bool_mask_rank3() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 3, DType::Bool(BoolStore::Native))
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            mask: Tensor<3, Bool>,
        ) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let hide = mask.bool_not().unsqueeze::<4>();
                let hidden_rows = hide.clone().all_dim(3);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(hide),
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_kv_cache_burn_attention() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 2, DType::Bool(BoolStore::Native))
            .input_tensor("past_k", 4, DType::F32)
            .input_tensor("past_v", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            mask: Tensor<2, Bool>,
            past_k: Tensor<4>,
            past_v: Tensor<4>,
        ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
            let (output, present_k, present_v) = {
                let q = query;
                let k = key;
                let v = value;
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let hide = mask.bool_not().unsqueeze::<4>();
                let hidden_rows = hide.clone().all_dim(3);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(hide),
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output, present_k, present_v)
            };
            (output, present_k, present_v)
        }
        ");
    }

    #[test]
    fn test_attention_with_past_present_kv() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("bias", 4, DType::F32) // slot 3
            .input_tensor("past_k", 4, DType::F32) // slot 4
            .input_tensor("past_v", 4, DType::F32) // slot 5
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            bias: Tensor<4>,
            past_k: Tensor<4>,
            past_v: Tensor<4>,
        ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
            let (output, present_k, present_v) = {
                let q = query;
                let k = key;
                let v = value;
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let score_bias = bias;
                let hidden_rows = score_bias.clone().max_dim(3).equal_elem(f32::NEG_INFINITY);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    Some(score_bias),
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output, present_k, present_v)
            };
            (output, present_k, present_v)
        }
        ");
    }

    #[test]
    fn test_attention_qk_output_mode_matmul_plus_mask() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::MatmulPlusAttentionMask,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 4, DType::F32)
            .input_tensor("past_k", 4, DType::F32)
            .input_tensor("past_v", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .output_tensor("qk_output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            mask: Tensor<4>,
            past_k: Tensor<4>,
            past_v: Tensor<4>,
        ) -> (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let score_bias = {
                    let shape = [q.dims()[0], q.dims()[1], q.dims()[2], k.dims()[2]];
                    let score_bias = Tensor::<4>::zeros(shape, (&self.device, q.dtype()));
                    let score_bias = score_bias + mask.expand(shape);
                    score_bias
                };
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let qk = qk + score_bias.clone();
                let qk_output = qk.clone();
                let scores = softmax(qk, 3);
                let scores = {
                    let hidden_rows = score_bias.max_dim(3).equal_elem(f32::NEG_INFINITY);
                    let dims = scores.dims();
                    scores.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                let output = scores.matmul(v);
                (output, present_k, present_v, qk_output)
            };
            (output, present_k, present_v, qk_output)
        }
        ");
    }

    #[test]
    fn test_attention_qk_output_mode_after_softcap() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::MatmulAfterSoftcap,
            scale: None,
            softcap: 30.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("bias", 4, DType::F32)
            .input_tensor("past_k", 4, DType::F32)
            .input_tensor("past_v", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .output_tensor("qk_output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            bias: Tensor<4>,
            past_k: Tensor<4>,
            past_v: Tensor<4>,
        ) -> (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let score_bias = {
                    let shape = [q.dims()[0], q.dims()[1], q.dims()[2], k.dims()[2]];
                    let score_bias = Tensor::<4>::zeros(shape, (&self.device, q.dtype()));
                    let score_bias = score_bias + bias.expand(shape);
                    score_bias
                };
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let qk = (qk * 0.03333333333333333f64).tanh() * 30f64;
                let qk_output = qk.clone();
                let qk = qk + score_bias.clone();
                let scores = softmax(qk, 3);
                let scores = {
                    let hidden_rows = score_bias.max_dim(3).equal_elem(f32::NEG_INFINITY);
                    let dims = scores.dims();
                    scores.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                let output = scores.matmul(v);
                (output, present_k, present_v, qk_output)
            };
            (output, present_k, present_v, qk_output)
        }
        ");
    }

    #[test]
    fn test_attention_qk_output_mode_after_softmax() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::MatmulAfterSoftmax,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("bias", 4, DType::F32)
            .input_tensor("past_k", 4, DType::F32)
            .input_tensor("past_v", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .output_tensor("qk_output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            bias: Tensor<4>,
            past_k: Tensor<4>,
            past_v: Tensor<4>,
        ) -> (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let score_bias = {
                    let shape = [q.dims()[0], q.dims()[1], q.dims()[2], k.dims()[2]];
                    let score_bias = Tensor::<4>::zeros(shape, (&self.device, q.dtype()));
                    let score_bias = score_bias + bias.expand(shape);
                    score_bias
                };
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let qk = qk + score_bias.clone();
                let scores = softmax(qk, 3);
                let scores = {
                    let hidden_rows = score_bias.max_dim(3).equal_elem(f32::NEG_INFINITY);
                    let dims = scores.dims();
                    scores.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                let qk_output = scores.clone();
                let output = scores.matmul(v);
                (output, present_k, present_v, qk_output)
            };
            (output, present_k, present_v, qk_output)
        }
        ");
    }

    #[test]
    fn test_attention_with_int_mask() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 4, DType::I64)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            mask: Tensor<4, Int>,
        ) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let score_bias = mask.float().cast(burn::tensor::DType::F32);
                let hidden_rows = score_bias.clone().max_dim(3).equal_elem(f32::NEG_INFINITY);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    Some(score_bias),
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_causal_with_mask() {
        // Per ONNX spec, is_causal masks scores above the diagonal regardless of
        // attn_mask, so both apply: the mask as a bias, the causal part as a mask.
        let config = AttentionConfig {
            is_causal: true,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            mask: Tensor<4>,
        ) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let groups = q.dims()[1] / k.dims()[1];
                let (k, v) = if groups > 1 {
                    let [batch, kv_heads, k_len, k_dim] = k.dims();
                    let v_dim = v.dims()[3];
                    (
                        k
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, k_dim])
                            .reshape([batch, kv_heads * groups, k_len, k_dim]),
                        v
                            .unsqueeze_dim::<5>(2)
                            .expand([batch, kv_heads, groups, k_len, v_dim])
                            .reshape([batch, kv_heads * groups, k_len, v_dim]),
                    )
                } else {
                    (k, v)
                };
                let hide = {
                    let [batch, heads, q_len, _] = q.dims();
                    let k_len = k.dims()[2];
                    let rows = Tensor::<
                        1,
                        Int,
                    >::arange(0..q_len as i64, (&self.device, burn::tensor::DType::I64))
                        .reshape([q_len, 1])
                        .expand([q_len, k_len]);
                    let cols = Tensor::<
                        1,
                        Int,
                    >::arange(0..k_len as i64, (&self.device, burn::tensor::DType::I64))
                        .reshape([1, k_len])
                        .expand([q_len, k_len]);
                    cols.greater(rows).unsqueeze::<4>().expand([batch, heads, q_len, k_len])
                };
                let score_bias = mask;
                let hidden_rows = {
                    let shape = [q.dims()[0], q.dims()[1], q.dims()[2], k.dims()[2]];
                    score_bias
                        .clone()
                        .expand(shape)
                        .mask_fill(hide.clone().expand(shape), f32::NEG_INFINITY)
                        .max_dim(3)
                        .equal_elem(f32::NEG_INFINITY)
                };
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(hide),
                    Some(score_bias),
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output,)
            };
            output
        }
        ");
    }

    fn causal_config() -> AttentionConfig {
        AttentionConfig {
            is_causal: true,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        }
    }

    #[test]
    fn test_attention_native_causal_equal_static_lengths() {
        // Q and K equally long with equal head counts: burn's own is_causal agrees
        // with ONNX and no K/V repeat is needed.
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor_shape("query", vec![1, 2, 3, 4], DType::F32)
            .input_tensor_shape("key", vec![1, 2, 3, 4], DType::F32)
            .input_tensor_shape("value", vec![1, 2, 3, 4], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(causal_config())
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, query: Tensor<4>, key: Tensor<4>, value: Tensor<4>) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: true,
                    },
                );
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_equal_static_lengths_with_mask_builds_causal() {
        // The row guard has to see the causal mask too, so it is not left to burn.
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor_shape("query", vec![1, 2, 3, 4], DType::F32)
            .input_tensor_shape("key", vec![1, 2, 3, 4], DType::F32)
            .input_tensor_shape("value", vec![1, 2, 3, 4], DType::F32)
            .input_tensor("attn_mask", 2, DType::Bool(BoolStore::Native))
            .output_tensor("output", 4, DType::F32)
            .config(causal_config())
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            attn_mask: Tensor<2, Bool>,
        ) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let causal = {
                    let [batch, heads, q_len, _] = q.dims();
                    let k_len = k.dims()[2];
                    let rows = Tensor::<
                        1,
                        Int,
                    >::arange(0..q_len as i64, (&self.device, burn::tensor::DType::I64))
                        .reshape([q_len, 1])
                        .expand([q_len, k_len]);
                    let cols = Tensor::<
                        1,
                        Int,
                    >::arange(0..k_len as i64, (&self.device, burn::tensor::DType::I64))
                        .reshape([1, k_len])
                        .expand([q_len, k_len]);
                    cols.greater(rows).unsqueeze::<4>().expand([batch, heads, q_len, k_len])
                };
                let hide = attn_mask
                    .bool_not()
                    .unsqueeze::<4>()
                    .expand(causal.dims())
                    .bool_or(causal);
                let hidden_rows = hide.clone().all_dim(3);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(hide),
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_explicit_causal_with_bool_mask() {
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor_shape("query", vec![2, 2, 2, 4], DType::F32)
            .input_tensor_shape("key", vec![2, 2, 3, 4], DType::F32)
            .input_tensor_shape("value", vec![2, 2, 3, 4], DType::F32)
            .input_tensor("padding", 4, DType::Bool(BoolStore::Native))
            .output_tensor("output", 4, DType::F32)
            .config(causal_config())
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            padding: Tensor<4, Bool>,
        ) -> Tensor<4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let causal = {
                    let [batch, heads, q_len, _] = q.dims();
                    let k_len = k.dims()[2];
                    let rows = Tensor::<
                        1,
                        Int,
                    >::arange(0..q_len as i64, (&self.device, burn::tensor::DType::I64))
                        .reshape([q_len, 1])
                        .expand([q_len, k_len]);
                    let cols = Tensor::<
                        1,
                        Int,
                    >::arange(0..k_len as i64, (&self.device, burn::tensor::DType::I64))
                        .reshape([1, k_len])
                        .expand([q_len, k_len]);
                    cols.greater(rows).unsqueeze::<4>().expand([batch, heads, q_len, k_len])
                };
                let hide = padding.bool_not().expand(causal.dims()).bool_or(causal);
                let hidden_rows = hide.clone().all_dim(3);
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(hide),
                    None,
                    burn::tensor::ops::AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
                let output = {
                    let dims = output.dims();
                    output.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_custom_causal_with_mask() {
        let mut config = causal_config();
        config.softcap = 2.0;
        config.qk_matmul_output_mode = AttentionQkMatmulOutputMode::MatmulPlusAttentionMask;
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor_shape("query", vec![1, 2, 2, 4], DType::F32)
            .input_tensor_shape("key", vec![1, 2, 3, 4], DType::F32)
            .input_tensor_shape("value", vec![1, 2, 3, 4], DType::F32)
            .input_tensor("bias", 2, DType::F32)
            .input_tensor("past_key", 4, DType::F32)
            .input_tensor("past_value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_key", 4, DType::F32)
            .output_tensor("present_value", 4, DType::F32)
            .output_tensor("qk_output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<4>,
            key: Tensor<4>,
            value: Tensor<4>,
            bias: Tensor<2>,
            past_key: Tensor<4>,
            past_value: Tensor<4>,
        ) -> (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>) {
            let (output, present_key, present_value, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let new_len = k.dims()[2];
                let present_key = Tensor::cat([past_key, k].to_vec(), 2);
                let k = present_key.clone();
                let present_value = Tensor::cat([past_value, v].to_vec(), 2);
                let v = present_value.clone();
                let past_len = k.dims()[2] - new_len;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let score_bias = {
                    let shape = [q.dims()[0], q.dims()[1], q.dims()[2], k.dims()[2]];
                    let score_bias = Tensor::<4>::zeros(shape, (&self.device, q.dtype()));
                    let score_bias = score_bias + bias.unsqueeze::<4>().expand(shape);
                    let score_bias = score_bias
                        .mask_fill(
                            {
                                let [batch, heads, q_len, _] = q.dims();
                                let k_len = k.dims()[2];
                                let rows = Tensor::<
                                    1,
                                    Int,
                                >::arange(
                                        0..q_len as i64,
                                        (&self.device, burn::tensor::DType::I64),
                                    )
                                    .reshape([q_len, 1])
                                    .add_scalar(past_len as i64)
                                    .expand([q_len, k_len]);
                                let cols = Tensor::<
                                    1,
                                    Int,
                                >::arange(
                                        0..k_len as i64,
                                        (&self.device, burn::tensor::DType::I64),
                                    )
                                    .reshape([1, k_len])
                                    .expand([q_len, k_len]);
                                cols.greater(rows)
                                    .unsqueeze::<4>()
                                    .expand([batch, heads, q_len, k_len])
                            },
                            f32::NEG_INFINITY,
                        );
                    score_bias
                };
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let qk = (qk * 0.5f64).tanh() * 2f64;
                let qk = qk + score_bias.clone();
                let qk_output = qk.clone();
                let scores = softmax(qk, 3);
                let scores = {
                    let hidden_rows = score_bias.max_dim(3).equal_elem(f32::NEG_INFINITY);
                    let dims = scores.dims();
                    scores.mask_fill(hidden_rows.expand(dims), 0.0)
                };
                let output = scores.matmul(v);
                (output, present_key, present_value, qk_output)
            };
            (output, present_key, present_value, qk_output)
        }
        ");
    }
}
