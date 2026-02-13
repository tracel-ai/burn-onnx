use super::prelude::*;
use onnx_ir::node::attention::AttentionQkMatmulOutputMode;

/// Whether this attention node can use `burn::tensor::module::attention()`.
///
/// Burn's attention natively supports scale, softcap, is_causal, bool masks,
/// and float additive biases. The only feature requiring custom codegen is
/// qk_matmul intermediate output.
fn use_burn_attention(node: &onnx_ir::attention::AttentionNode) -> bool {
    node.outputs.get(3).is_none()
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

    let mut body = TokenStream::new();

    body.extend(quote! {
        let q = #q;
        let k = #k;
        let v = #v;
    });

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

    // KV cache concatenation
    if past_kv {
        let past_k = scope.arg(node.inputs.get(4).unwrap());
        let past_v = scope.arg(node.inputs.get(5).unwrap());
        let present_k = arg_to_ident(node.outputs.get(1).unwrap());
        let present_v = arg_to_ident(node.outputs.get(2).unwrap());

        body.extend(quote! {
            let #present_k = Tensor::cat([#past_k, k].to_vec(), 2);
            let k = #present_k.clone();
            let #present_v = Tensor::cat([#past_v, v].to_vec(), 2);
            let v = #present_v.clone();
        });
    }

    // Build AttentionOptions
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
    let is_causal = node.config.is_causal;
    let options = quote! {
        burn::tensor::ops::AttentionOptions {
            scale: #scale_tokens,
            softcap: #softcap_tokens,
            is_causal: #is_causal,
        }
    };

    // Mask handling:
    // - Bool masks -> `mask` parameter (inverted: ONNX attend=true -> Burn masked=true)
    // - Float/int masks -> `attn_bias` parameter (additive bias)
    // - Causal masking is handled natively by the backend via is_causal
    let mask_input = if !node.config.is_causal {
        node.inputs.get(3).filter(|a| !a.is_optional())
    } else {
        None
    };

    let (mask_tokens, bias_tokens) = if let Some(mask_input) = mask_input {
        let mask_arg = scope.arg(mask_input);
        match &mask_input.ty {
            ArgType::Tensor(t) if t.dtype.is_bool() => {
                let mask = match t.rank {
                    2 => quote! { #mask_arg.bool_not().unsqueeze::<4>() },
                    3 => quote! { #mask_arg.bool_not().unsqueeze_dim::<4>(1) },
                    4 => quote! { #mask_arg.bool_not() },
                    _ => panic!("Attention mask must be rank 2, 3, or 4"),
                };
                (quote! { Some(#mask) }, quote! { None })
            }
            ArgType::Tensor(t) if t.dtype.is_float() => {
                let bias = match t.rank {
                    2 => quote! { #mask_arg.unsqueeze::<4>() },
                    3 => quote! { #mask_arg.unsqueeze_dim::<4>(1) },
                    4 => mask_arg,
                    _ => panic!("Attention bias must be rank 2, 3, or 4"),
                };
                (quote! { None }, quote! { Some(#bias) })
            }
            ArgType::Tensor(t) if t.dtype.is_int() || t.dtype.is_uint() => {
                let bias = match t.rank {
                    2 => quote! { #mask_arg.float().unsqueeze::<4>() },
                    3 => quote! { #mask_arg.float().unsqueeze_dim::<4>(1) },
                    4 => quote! { #mask_arg.float() },
                    _ => panic!("Attention bias must be rank 2, 3, or 4"),
                };
                (quote! { None }, quote! { Some(#bias) })
            }
            _ => panic!("Unsupported attention mask type"),
        }
    } else {
        (quote! { None }, quote! { None })
    };

    let attention_call = quote! {
        let #output_y = burn::tensor::module::attention(q, k, v, #mask_tokens, #bias_tokens, #options);
    };
    body.extend(attention_call);
    body.extend(reshape_output);

    // Build output tuple
    let mut output_names = vec![output_y.clone()];
    if present_kv {
        output_names.push(arg_to_ident(node.outputs.get(1).unwrap()));
        output_names.push(arg_to_ident(node.outputs.get(2).unwrap()));
    }
    let output = quote! { (#(#output_names,)*) };

    quote! {
        let #output = {
            #body
            #output
        };
    }
}

/// Fallback codegen for qk_matmul intermediate output, which is not
/// supported by burn's attention API.
fn forward_custom(
    node: &onnx_ir::attention::AttentionNode,
    scope: &mut ScopeAtPosition<'_>,
) -> TokenStream {
    let q = scope.arg(node.inputs.first().unwrap());
    let k = scope.arg(node.inputs.get(1).unwrap());
    let v = scope.arg(node.inputs.get(2).unwrap());
    let output_y = arg_to_ident(node.outputs.first().unwrap());

    let past_kv = match (node.inputs.get(4), node.inputs.get(5)) {
        (Some(_), Some(_)) => true,
        (None, None) => false,
        _ => panic!("Attention: past_key and past_value must be used together."),
    };
    let present_kv = match (node.outputs.get(1), node.outputs.get(2)) {
        (Some(_), Some(_)) => true,
        (None, None) => false,
        _ => panic!("Attention: present_key and present_value must be used together."),
    };

    let rank = match &node.inputs.first().unwrap().ty {
        onnx_ir::ir::ArgType::Tensor(t) => t.rank,
        _ => panic!("Expected tensor input for Q"),
    };

    let mut body = TokenStream::new();

    body.extend(quote! {
        let q = #q;
        let k = #k;
        let v = #v;
    });

    let scale = node.config.scale.map(|scale| {
        let scale = scale.sqrt();
        quote! {
            let scale = #scale;
        }
    });

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
        body.extend(scale.unwrap_or_else(|| {
            quote! {
                let scale = (1.0 / (head_size as f64).sqrt()).sqrt();
            }
        }));

        reshape_output = quote! {
            let #output_y = #output_y.permute([0, 2, 1, 3]).reshape([batch_size as i32, q_sequence_length as i32, -1]);
        };
    } else {
        body.extend(scale.unwrap_or_else(|| {
            quote! {
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
            }
        }));
    }

    if past_kv && present_kv {
        let past_k = scope.arg(node.inputs.get(4).unwrap());
        let past_v = scope.arg(node.inputs.get(5).unwrap());
        let present_k = arg_to_ident(node.outputs.get(1).unwrap());
        let present_v = arg_to_ident(node.outputs.get(2).unwrap());

        body.extend(quote! {
            let #present_k = Tensor::cat([#past_k, k].to_vec(), 2);
            let k = #present_k.clone();
            let #present_v = Tensor::cat([#past_v, v].to_vec(), 2);
            let v = #present_v.clone();
        });
    } else if past_kv != present_kv {
        panic!("Attention: past_[key,value] and present_[key,value] must be used together.")
    }

    if node.inputs.get(3).is_some_and(|a| !a.is_optional()) || node.config.is_causal {
        body.extend(quote! {
            let q_dims = q.dims();
            let k_dims = k.dims();
        });
    }

    let qk = quote! { qk };
    let attn_mask_shape = quote! {{
        let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
        [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
    }};

    let mut attn_mask = if let Some(mask_input) = node.inputs.get(3).filter(|a| !a.is_optional()) {
        let mask_arg = scope.arg(mask_input);
        let mask = match &mask_input.ty {
            onnx_ir::ir::ArgType::Tensor(t) => match &t.dtype {
                dtype if dtype.is_int() || dtype.is_uint() => {
                    quote! { #mask_arg.float() }
                }
                dtype if dtype.is_float() => mask_arg,
                dtype if dtype.is_bool() => {
                    quote! {{
                        let float_mask = Tensor::<B, 2>::zeros([shape[2], shape[3]], &#mask_arg.device());
                        float_mask.mask_fill(#mask_arg.bool_not(), f32::NEG_INFINITY)
                    }}
                }
                _ => panic!("Unsupported mask dtype"),
            },
            _ => panic!("Attention mask must be a tensor"),
        };

        quote! {
            let shape = #attn_mask_shape;
            let #qk = #qk + #mask.expand::<4, _>(shape);
        }
    } else {
        quote! {}
    };

    if node.config.is_causal {
        attn_mask = quote! {
            let #qk = {
                let shape = #attn_mask_shape;
                let mask = Tensor::<B, 2>::ones([shape[2], shape[3]], &#qk.device());
                let mask = mask.tril(0).bool().bool_not();
                let float_mask = Tensor::<B, 2>::zeros([shape[2], shape[3]], &mask.device()).mask_fill(mask, f32::NEG_INFINITY);
                #qk + float_mask.expand::<4, _>(shape)
            };
        };
    }

    let capped = quote! { capped };
    let (mut qk_matmul_a, mut qk_matmul_b, mut qk_matmul_c, mut qk_matmul_d) =
        (quote! {}, quote! {}, quote! {}, quote! {});
    if let Some(out_arg) = node.outputs.get(3) {
        let out = arg_to_ident(out_arg);
        match node.config.qk_matmul_output_mode {
            AttentionQkMatmulOutputMode::Matmul => {
                qk_matmul_a = quote! {
                    let #out = #qk.clone();
                };
            }
            AttentionQkMatmulOutputMode::MatmulPlusAttentionMask => {
                qk_matmul_b = quote! {
                    let #out = #qk.clone();
                };
            }
            AttentionQkMatmulOutputMode::MatmulAfterSoftcap => {
                qk_matmul_c = quote! {
                    let #out = #capped.clone();
                };
            }
            AttentionQkMatmulOutputMode::MatmulAfterSoftmax => {
                qk_matmul_d = quote! {
                    let #out = scores.clone();
                };
            }
        }
    }

    let softcap = if node.config.softcap != 0.0 {
        let softcap = node.config.softcap;
        let inv_softcap = 1.0 / softcap;
        quote! {
            let #capped = {
                let score = #qk * #inv_softcap;
                let score = score.tanh();
                score * #softcap
            };
            #qk_matmul_c
        }
    } else {
        quote! {
            let #capped = #qk;
        }
    };

    let mut output_names = vec![output_y.clone()];
    if present_kv {
        output_names.push(arg_to_ident(node.outputs.get(1).unwrap()));
        output_names.push(arg_to_ident(node.outputs.get(2).unwrap()));
    }
    if node.outputs.get(3).is_some() {
        output_names.push(arg_to_ident(node.outputs.get(3).unwrap()));
    }
    let output = quote! { (#(#output_names,)*) };

    quote! {
        let #output = {
            #body

            let q_scaled = q * scale;
            let k_scaled = k * scale;
            let k_transpose = k_scaled.transpose();
            let #qk = q_scaled.matmul(k_transpose);
            #qk_matmul_a
            #attn_mask
            #qk_matmul_b
            #softcap
            let scores = softmax(#capped, 3);
            #qk_matmul_d
            let #output_y = scores.matmul(v);
            #reshape_output
            #output
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
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
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
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
                    burn::tensor::ops::AttentionOptions {
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
        pub fn forward(
            &self,
            query: Tensor<B, 3>,
            key: Tensor<B, 3>,
            value: Tensor<B, 3>,
        ) -> Tensor<B, 3> {
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
                    burn::tensor::ops::AttentionOptions {
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
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
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
                    burn::tensor::ops::AttentionOptions {
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
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    Some(mask),
                    burn::tensor::ops::AttentionOptions {
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
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
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
                    burn::tensor::ops::AttentionOptions {
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
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
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
                    burn::tensor::ops::AttentionOptions {
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
            .input_tensor("mask", 2, DType::Bool)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 2, Bool>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(mask.bool_not().unsqueeze::<4>()),
                    None,
                    burn::tensor::ops::AttentionOptions {
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
            .input_tensor("mask", 3, DType::Bool)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 3, Bool>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(mask.bool_not().unsqueeze_dim::<4>(1)),
                    None,
                    burn::tensor::ops::AttentionOptions {
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
            .input_tensor("mask", 2, DType::Bool)
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
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 2, Bool>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v) = {
                let q = query;
                let k = key;
                let v = value;
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    Some(mask.bool_not().unsqueeze::<4>()),
                    None,
                    burn::tensor::ops::AttentionOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
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
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            bias: Tensor<B, 4>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v) = {
                let q = query;
                let k = key;
                let v = value;
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    Some(bias),
                    burn::tensor::ops::AttentionOptions {
                        scale: None,
                        softcap: None,
                        is_causal: false,
                    },
                );
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
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 4>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let q_dims = q.dims();
                let k_dims = k.dims();
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let shape = {
                    let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
                    [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
                };
                let qk = qk + mask.expand::<4, _>(shape);
                let qk_output = qk.clone();
                let capped = qk;
                let scores = softmax(capped, 3);
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
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            bias: Tensor<B, 4>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let q_dims = q.dims();
                let k_dims = k.dims();
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let shape = {
                    let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
                    [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
                };
                let qk = qk + bias.expand::<4, _>(shape);
                let capped = {
                    let score = qk * 0.03333333333333333f64;
                    let score = score.tanh();
                    score * 30f64
                };
                let qk_output = capped.clone();
                let scores = softmax(capped, 3);
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
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            bias: Tensor<B, 4>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
                let q_dims = q.dims();
                let k_dims = k.dims();
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let shape = {
                    let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
                    [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
                };
                let qk = qk + bias.expand::<4, _>(shape);
                let capped = qk;
                let scores = softmax(capped, 3);
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
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 4, Int>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let output = burn::tensor::module::attention(
                    q,
                    k,
                    v,
                    None,
                    Some(mask.float()),
                    burn::tensor::ops::AttentionOptions {
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
    fn test_attention_causal_with_mask_ignores_mask() {
        // Per ONNX spec: "is_causal masks scores above the diagonal, regardless of attn_mask"
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
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
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
                    burn::tensor::ops::AttentionOptions {
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
}
