use super::prelude::*;
use crate::burn::codegen::f32_to_tokens;
use onnx_ir::ir::ArgType;

impl NodeCodegen for onnx_ir::pad::PadNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let pads_expr: TokenStream = match &self.config.pads {
            onnx_ir::pad::PadInput::Static(pads) => {
                let pads = pads
                    .iter()
                    .map(|(before, after)| quote! { (#before, #after) });
                quote! { [#(#pads),*] }
            }
            onnx_ir::pad::PadInput::Runtime {
                input: runtime_ref,
                axes,
            } => {
                let data_arg = self.inputs.first().unwrap();
                let input_rank = match &data_arg.ty {
                    ArgType::Tensor(t) => t.rank,
                    other => panic!("Pad: data input must be a tensor, got {other:?}"),
                };
                runtime_pads_expr(
                    scope,
                    &self.inputs,
                    &self.inputs[runtime_ref.input_index],
                    axes,
                    input_rank,
                )
            }
        };

        let pad_mode = match &self.config.mode {
            onnx_ir::pad::PadMode::Constant => {
                let constant_value =
                    constant_value_expr(scope, &self.inputs, &self.config.constant_value);
                quote! { burn::tensor::ops::PadMode::Constant(#constant_value) }
            }
            onnx_ir::pad::PadMode::Reflect => quote! { burn::tensor::ops::PadMode::Reflect },
            onnx_ir::pad::PadMode::Edge => quote! { burn::tensor::ops::PadMode::Edge },
        };

        quote! {
            let #output = #input.pad(#pads_expr, #pad_mode);
        }
    }
}

/// Emit a forward-time expression that produces a pad array for
/// `Tensor::pad`. See [`onnx_ir::pad::PadInput`] for the index-space
/// contract (ONNX `pads` layout vs. burn's `(before, after)` per dim).
fn runtime_pads_expr(
    scope: &mut ScopeAtPosition<'_>,
    inputs: &[Argument],
    pads_arg: &Argument,
    axes: &Option<onnx_ir::pad::AxesInput>,
    input_rank: usize,
) -> TokenStream {
    let pads_value = scope.arg(pads_arg);
    // Runtime axes: pad-pair count only known at forward() time, so
    // bypass the static-specialization paths below.
    if let Some(onnx_ir::pad::AxesInput::Runtime(axes_ref)) = axes {
        let axes_arg = &inputs[axes_ref.input_index];
        let axes_value = scope.arg(axes_arg);
        return runtime_axes_scatter(&pads_value, pads_arg, &axes_value, axes_arg, input_rank);
    }

    let static_axes: Option<&[usize]> = match axes {
        Some(onnx_ir::pad::AxesInput::Static(a)) => Some(a.as_slice()),
        Some(onnx_ir::pad::AxesInput::Runtime(_)) => unreachable!("handled above"),
        None => None,
    };
    let expected_len = match static_axes {
        Some(a) => 2 * a.len(),
        None => 2 * input_rank,
    };

    let inline_pairs = |source: &TokenStream| match static_axes {
        None => full_rank_pairs(source, input_rank),
        Some(axes_vec) => axes_pairs(source, axes_vec, input_rank),
    };

    match &pads_arg.ty {
        ArgType::Shape(_) => inline_pairs(&pads_value),
        ArgType::Tensor(_) => {
            let to_vec = crate::burn::codegen::tensor_to_i64_vec(&pads_value);
            let pairs = inline_pairs(&quote! { __raw });
            quote! {
                {
                    let __raw: alloc::vec::Vec<i64> = #to_vec;
                    assert_eq!(
                        __raw.len(), #expected_len,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        #expected_len, __raw.len(),
                    );
                    #pairs
                }
            }
        }
        other => panic!("Pad: runtime pads input must be a tensor or shape, got {other:?}"),
    }
}

/// Emit forward()-time scatter for the case where both `pads` and
/// `axes` arrive at runtime, so no static specialization is possible.
fn runtime_axes_scatter(
    pads_value: &TokenStream,
    pads_arg: &Argument,
    axes_value: &TokenStream,
    axes_arg: &Argument,
    input_rank: usize,
) -> TokenStream {
    let pads_to_vec = match &pads_arg.ty {
        ArgType::Tensor(_) => crate::burn::codegen::tensor_to_i64_vec(pads_value),
        ArgType::Shape(_) => quote! {
            #pads_value.iter().copied().collect::<alloc::vec::Vec<i64>>()
        },
        other => {
            panic!("Pad: runtime pads input must be a tensor or shape, got {other:?}")
        }
    };
    let axes_to_vec = match &axes_arg.ty {
        ArgType::Tensor(_) => crate::burn::codegen::tensor_to_i64_vec(axes_value),
        ArgType::Shape(_) => quote! {
            #axes_value.iter().copied().collect::<alloc::vec::Vec<i64>>()
        },
        other => panic!("Pad: runtime axes input must be a tensor or shape, got {other:?}"),
    };
    let before_pad = pad_i64_to_usize_expr(quote! { __raw_pads[__i] }, quote! { __i });
    let after_pad = pad_i64_to_usize_expr(quote! { __raw_pads[__n + __i] }, quote! { __n + __i });
    quote! {
        {
            let __raw_pads: alloc::vec::Vec<i64> = #pads_to_vec;
            let __raw_axes: alloc::vec::Vec<i64> = #axes_to_vec;
            let __n = __raw_axes.len();
            assert_eq!(
                __raw_pads.len(), 2 * __n,
                "Pad: runtime pads length mismatch (expected 2 * axes.len() = {}, got {})",
                2 * __n, __raw_pads.len(),
            );
            let mut __pads: alloc::vec::Vec<(usize, usize)> =
                alloc::vec![(0usize, 0usize); #input_rank];
            let mut __seen: [bool; #input_rank] = [false; #input_rank];
            for __i in 0..__n {
                let __raw_axis = __raw_axes[__i];
                let __dim_signed = if __raw_axis < 0 {
                    __raw_axis + (#input_rank as i64)
                } else {
                    __raw_axis
                };
                assert!(
                    __dim_signed >= 0 && (__dim_signed as usize) < #input_rank,
                    "Pad: axis {} out of range for rank {}", __raw_axis, #input_rank,
                );
                let __dim = __dim_signed as usize;
                assert!(
                    !__seen[__dim],
                    "Pad: duplicate axis {} (normalized to dim {})", __raw_axis, __dim,
                );
                __seen[__dim] = true;
                __pads[__dim] = (#before_pad, #after_pad);
            }
            __pads
        }
    }
}

/// Inline `[(usize, usize); rank]` array literal picking `(before, after)`
/// pairs from `source` in ONNX layout (`source[i]`, `source[rank + i]`).
fn full_rank_pairs(source: &TokenStream, rank: usize) -> TokenStream {
    let pairs = (0..rank).map(|i| {
        let after = rank + i;
        let b = pad_i64_to_usize_expr(quote! { #source[#i] }, quote! { #i });
        let a = pad_i64_to_usize_expr(quote! { #source[#after] }, quote! { #after });
        quote! { (#b, #a) }
    });
    quote! { [#(#pairs),*] }
}

/// Inline `[(usize, usize); rank]` literal placing the `i`-th pair on
/// dimension `axes[i]` and `(0, 0)` on every unlisted dimension. `source`
/// supplies the runtime values in ONNX layout (`source[i]` is `before`
/// for `axes[i]`; `source[axes.len() + i]` is `after`).
fn axes_pairs(source: &TokenStream, axes: &[usize], rank: usize) -> TokenStream {
    let n_axes = axes.len();
    let pairs = (0..rank).map(|dim| match axes.iter().position(|&a| a == dim) {
        Some(slot) => {
            let after_slot = n_axes + slot;
            let b = pad_i64_to_usize_expr(quote! { #source[#slot] }, quote! { #slot });
            let a = pad_i64_to_usize_expr(quote! { #source[#after_slot] }, quote! { #after_slot });
            quote! { (#b, #a) }
        }
        None => quote! { (0usize, 0usize) },
    });
    quote! { [#(#pairs),*] }
}

fn constant_value_expr(
    scope: &mut ScopeAtPosition<'_>,
    inputs: &[Argument],
    cv: &onnx_ir::pad::ConstantValueInput,
) -> TokenStream {
    match cv {
        onnx_ir::pad::ConstantValueInput::Static(value) => f32_to_tokens(*value),
        onnx_ir::pad::ConstantValueInput::Runtime(runtime_ref) => {
            let arg = &inputs[runtime_ref.input_index];
            let value = scope.arg(arg);
            match &arg.ty {
                ArgType::Tensor(t) if t.rank == 0 => {
                    quote! { #value.into_scalar() }
                }
                ArgType::Tensor(t) => {
                    panic!(
                        "Pad: constant_value must be a scalar tensor (rank 0), got rank {}",
                        t.rank
                    )
                }
                ArgType::ScalarNative(_) => quote! { #value },
                ArgType::ScalarTensor(dtype) => on_device_to_native(quote! { #value }, dtype),
                ArgType::Shape(_) => panic!("Pad: constant_value cannot be a shape"),
            }
        }
    }
}

/// Checked i64-to-usize conversion emitted into generated code.
///
/// ONNX's static validator rejects negative pads at parse time, but a
/// runtime tensor can still carry them; surface the error loudly rather
/// than wrap to `usize::MAX - n` (which silently OOMs or panics deep in
/// burn's pad kernel).
fn pad_i64_to_usize_expr(value: TokenStream, idx: TokenStream) -> TokenStream {
    quote! {
        usize::try_from(#value).unwrap_or_else(|_| {
            panic!("Pad: negative pad value {} at index {}", #value, #idx)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::pad::{
        AxesInput, ConstantValueInput, PadConfig, PadInput, PadMode, PadNode, PadNodeBuilder,
    };

    fn create_pad_node(
        name: &str,
        pads: Vec<(usize, usize)>,
        constant_value: f32,
        mode: PadMode,
    ) -> PadNode {
        let config = PadConfig {
            pads: PadInput::Static(pads),
            constant_value: ConstantValueInput::Static(constant_value),
            mode,
        };

        PadNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_pad_constant_simple() {
        let node = create_pad_node("pad1", vec![(1, 1), (1, 1)], 0.0, PadMode::Constant);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input
                .pad(
                    [(1usize, 1usize), (1usize, 1usize)],
                    burn::tensor::ops::PadMode::Constant(0f32),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_constant_asymmetric() {
        let node = create_pad_node("pad1", vec![(0, 1), (2, 0)], 5.5, PadMode::Constant);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input
                .pad(
                    [(0usize, 1usize), (2usize, 0usize)],
                    burn::tensor::ops::PadMode::Constant(5.5f32),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_reflect() {
        let node = create_pad_node("pad1", vec![(1, 1), (1, 1)], 0.0, PadMode::Reflect);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input
                .pad([(1usize, 1usize), (1usize, 1usize)], burn::tensor::ops::PadMode::Reflect);
            output
        }
        ");
    }

    #[test]
    fn test_pad_edge() {
        let node = create_pad_node("pad1", vec![(1, 1), (1, 1)], 0.0, PadMode::Edge);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input
                .pad([(1usize, 1usize), (1usize, 1usize)], burn::tensor::ops::PadMode::Edge);
            output
        }
        ");
    }

    #[test]
    fn test_pad_constant_runtime_value() {
        let config = PadConfig {
            pads: PadInput::Static(vec![(1, 1), (1, 1)]),
            constant_value: ConstantValueInput::Runtime(RuntimeInputRef {
                name: "constant_value".to_string(),
                input_index: 1,
            }),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad1")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("constant_value", 0, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, constant_value: Tensor<0>) -> Tensor<2> {
            let output = input
                .pad(
                    [(1usize, 1usize), (1usize, 1usize)],
                    burn::tensor::ops::PadMode::Constant(constant_value.into_scalar()),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_constant_runtime_scalar() {
        let config = PadConfig {
            pads: PadInput::Static(vec![(1, 1), (1, 1)]),
            constant_value: ConstantValueInput::Runtime(RuntimeInputRef {
                name: "constant_value".to_string(),
                input_index: 1,
            }),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar("constant_value", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, constant_value: f32) -> Tensor<2> {
            let output = input
                .pad(
                    [(1usize, 1usize), (1usize, 1usize)],
                    burn::tensor::ops::PadMode::Constant(constant_value),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_constant_neg_infinity() {
        // Regression: `format!("{}_f32", -inf)` -> `-inf_f32`, not a
        // valid Rust literal. Covered by `f32_to_tokens` dispatch.
        let node = create_pad_node(
            "pad_inf",
            vec![(1, 1), (1, 1)],
            f32::NEG_INFINITY,
            PadMode::Constant,
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input
                .pad(
                    [(1usize, 1usize), (1usize, 1usize)],
                    burn::tensor::ops::PadMode::Constant(f32::NEG_INFINITY),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_runtime_pads_full_rank_tensor() {
        let config = PadConfig {
            pads: PadInput::Runtime {
                input: RuntimeInputRef::new("pads".to_string(), 1),
                axes: None,
            },
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad_rt")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("pads", 1, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<2>, pads: Tensor<1, Int>) -> Tensor<2> {
            let output = input
                .pad(
                    {
                        let __raw: alloc::vec::Vec<i64> = pads
                            .to_data()
                            .convert::<i64>()
                            .into_vec::<i64>()
                            .unwrap();
                        assert_eq!(
                            __raw.len(), 4usize,
                            "Pad: runtime pads length mismatch (expected {}, got {})", 4usize,
                            __raw.len(),
                        );
                        [
                            (
                                usize::try_from(__raw[0usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[0usize],
                                            0usize
                                        )
                                    }),
                                usize::try_from(__raw[2usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[2usize],
                                            2usize
                                        )
                                    }),
                            ),
                            (
                                usize::try_from(__raw[1usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[1usize],
                                            1usize
                                        )
                                    }),
                                usize::try_from(__raw[3usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[3usize],
                                            3usize
                                        )
                                    }),
                            ),
                        ]
                    },
                    burn::tensor::ops::PadMode::Constant(0f32),
                );
            output
        }
        "#);
    }

    #[test]
    fn test_pad_runtime_pads_full_rank_shape() {
        // Shape input → inline-array codegen (no Vec allocation).
        let config = PadConfig {
            pads: PadInput::Runtime {
                input: RuntimeInputRef::new("pads".to_string(), 1),
                axes: None,
            },
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad_rt_shape")
            .input_tensor("input", 2, DType::F32)
            .input_shape("pads", 4)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<2>, pads: [i64; 4]) -> Tensor<2> {
            let output = input
                .pad(
                    [
                        (
                            usize::try_from(pads[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", pads[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(pads[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", pads[2usize],
                                        2usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(pads[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", pads[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(pads[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", pads[3usize],
                                        3usize
                                    )
                                }),
                        ),
                    ],
                    burn::tensor::ops::PadMode::Constant(0f32),
                );
            output
        }
        "#);
    }

    #[test]
    fn test_pad_runtime_pads_with_static_axes_tensor() {
        let config = PadConfig {
            pads: PadInput::Runtime {
                input: RuntimeInputRef::new("pads".to_string(), 1),
                axes: Some(AxesInput::Static(vec![2, 0])),
            },
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad_rt_axes")
            .input_tensor("input", 4, DType::F32)
            .input_tensor("pads", 1, DType::I64)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<4>, pads: Tensor<1, Int>) -> Tensor<4> {
            let output = input
                .pad(
                    {
                        let __raw: alloc::vec::Vec<i64> = pads
                            .to_data()
                            .convert::<i64>()
                            .into_vec::<i64>()
                            .unwrap();
                        assert_eq!(
                            __raw.len(), 4usize,
                            "Pad: runtime pads length mismatch (expected {}, got {})", 4usize,
                            __raw.len(),
                        );
                        [
                            (
                                usize::try_from(__raw[1usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[1usize],
                                            1usize
                                        )
                                    }),
                                usize::try_from(__raw[3usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[3usize],
                                            3usize
                                        )
                                    }),
                            ),
                            (0usize, 0usize),
                            (
                                usize::try_from(__raw[0usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[0usize],
                                            0usize
                                        )
                                    }),
                                usize::try_from(__raw[2usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[2usize],
                                            2usize
                                        )
                                    }),
                            ),
                            (0usize, 0usize),
                        ]
                    },
                    burn::tensor::ops::PadMode::Constant(0f32),
                );
            output
        }
        "#);
    }

    #[test]
    fn test_pad_runtime_pads_with_static_axes_shape() {
        let config = PadConfig {
            pads: PadInput::Runtime {
                input: RuntimeInputRef::new("pads".to_string(), 1),
                axes: Some(AxesInput::Static(vec![2, 0])),
            },
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad_rt_axes_shape")
            .input_tensor("input", 4, DType::F32)
            .input_shape("pads", 4)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<4>, pads: [i64; 4]) -> Tensor<4> {
            let output = input
                .pad(
                    [
                        (
                            usize::try_from(pads[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", pads[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(pads[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", pads[3usize],
                                        3usize
                                    )
                                }),
                        ),
                        (0usize, 0usize),
                        (
                            usize::try_from(pads[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", pads[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(pads[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", pads[2usize],
                                        2usize
                                    )
                                }),
                        ),
                        (0usize, 0usize),
                    ],
                    burn::tensor::ops::PadMode::Constant(0f32),
                );
            output
        }
        "#);
    }

    #[test]
    fn test_pad_runtime_pads_reflect() {
        let config = PadConfig {
            pads: PadInput::Runtime {
                input: RuntimeInputRef::new("pads".to_string(), 1),
                axes: None,
            },
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Reflect,
        };
        let node = PadNodeBuilder::new("pad_rt_reflect")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("pads", 1, DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<3>, pads: Tensor<1, Int>) -> Tensor<3> {
            let output = input
                .pad(
                    {
                        let __raw: alloc::vec::Vec<i64> = pads
                            .to_data()
                            .convert::<i64>()
                            .into_vec::<i64>()
                            .unwrap();
                        assert_eq!(
                            __raw.len(), 6usize,
                            "Pad: runtime pads length mismatch (expected {}, got {})", 6usize,
                            __raw.len(),
                        );
                        [
                            (
                                usize::try_from(__raw[0usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[0usize],
                                            0usize
                                        )
                                    }),
                                usize::try_from(__raw[3usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[3usize],
                                            3usize
                                        )
                                    }),
                            ),
                            (
                                usize::try_from(__raw[1usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[1usize],
                                            1usize
                                        )
                                    }),
                                usize::try_from(__raw[4usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[4usize],
                                            4usize
                                        )
                                    }),
                            ),
                            (
                                usize::try_from(__raw[2usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[2usize],
                                            2usize
                                        )
                                    }),
                                usize::try_from(__raw[5usize])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw[5usize],
                                            5usize
                                        )
                                    }),
                            ),
                        ]
                    },
                    burn::tensor::ops::PadMode::Reflect,
                );
            output
        }
        "#);
    }

    #[test]
    fn test_pad_runtime_pads_with_shape_typed_runtime_axes() {
        // axes typed as Shape(N) (e.g. Concat'd Shape outputs) selects
        // the `[i64; N]` indexing branch of `axes_to_vec`.
        let config = PadConfig {
            pads: PadInput::Runtime {
                input: RuntimeInputRef::new("pads".to_string(), 1),
                axes: Some(AxesInput::Runtime(RuntimeInputRef::new(
                    "axes".to_string(),
                    2,
                ))),
            },
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad_rt_shape_axes")
            .input_tensor("input", 4, DType::F32)
            .input_tensor("pads", 1, DType::I64)
            .input_shape("axes", 2)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            input: Tensor<4>,
            pads: Tensor<1, Int>,
            axes: [i64; 2],
        ) -> Tensor<4> {
            let output = input
                .pad(
                    {
                        let __raw_pads: alloc::vec::Vec<i64> = pads
                            .to_data()
                            .convert::<i64>()
                            .into_vec::<i64>()
                            .unwrap();
                        let __raw_axes: alloc::vec::Vec<i64> = axes
                            .iter()
                            .copied()
                            .collect::<alloc::vec::Vec<i64>>();
                        let __n = __raw_axes.len();
                        assert_eq!(
                            __raw_pads.len(), 2 * __n,
                            "Pad: runtime pads length mismatch (expected 2 * axes.len() = {}, got {})",
                            2 * __n, __raw_pads.len(),
                        );
                        let mut __pads: alloc::vec::Vec<(usize, usize)> = alloc::vec![
                            (0usize, 0usize); 4usize
                        ];
                        let mut __seen: [bool; 4usize] = [false; 4usize];
                        for __i in 0..__n {
                            let __raw_axis = __raw_axes[__i];
                            let __dim_signed = if __raw_axis < 0 {
                                __raw_axis + (4usize as i64)
                            } else {
                                __raw_axis
                            };
                            assert!(
                                __dim_signed >= 0 && (__dim_signed as usize) < 4usize,
                                "Pad: axis {} out of range for rank {}", __raw_axis, 4usize,
                            );
                            let __dim = __dim_signed as usize;
                            assert!(
                                ! __seen[__dim], "Pad: duplicate axis {} (normalized to dim {})",
                                __raw_axis, __dim,
                            );
                            __seen[__dim] = true;
                            __pads[__dim] = (
                                usize::try_from(__raw_pads[__i])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw_pads[__i],
                                            __i
                                        )
                                    }),
                                usize::try_from(__raw_pads[__n + __i])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw_pads[__n +
                                            __i], __n + __i
                                        )
                                    }),
                            );
                        }
                        __pads
                    },
                    burn::tensor::ops::PadMode::Constant(0f32),
                );
            output
        }
        "#);
    }

    #[test]
    fn test_pad_runtime_pads_with_runtime_axes() {
        let config = PadConfig {
            pads: PadInput::Runtime {
                input: RuntimeInputRef::new("pads".to_string(), 1),
                axes: Some(AxesInput::Runtime(RuntimeInputRef::new(
                    "axes".to_string(),
                    2,
                ))),
            },
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad_rt_rt_axes")
            .input_tensor("input", 4, DType::F32)
            .input_tensor("pads", 1, DType::I64)
            .input_tensor("axes", 1, DType::I64)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            input: Tensor<4>,
            pads: Tensor<1, Int>,
            axes: Tensor<1, Int>,
        ) -> Tensor<4> {
            let output = input
                .pad(
                    {
                        let __raw_pads: alloc::vec::Vec<i64> = pads
                            .to_data()
                            .convert::<i64>()
                            .into_vec::<i64>()
                            .unwrap();
                        let __raw_axes: alloc::vec::Vec<i64> = axes
                            .to_data()
                            .convert::<i64>()
                            .into_vec::<i64>()
                            .unwrap();
                        let __n = __raw_axes.len();
                        assert_eq!(
                            __raw_pads.len(), 2 * __n,
                            "Pad: runtime pads length mismatch (expected 2 * axes.len() = {}, got {})",
                            2 * __n, __raw_pads.len(),
                        );
                        let mut __pads: alloc::vec::Vec<(usize, usize)> = alloc::vec![
                            (0usize, 0usize); 4usize
                        ];
                        let mut __seen: [bool; 4usize] = [false; 4usize];
                        for __i in 0..__n {
                            let __raw_axis = __raw_axes[__i];
                            let __dim_signed = if __raw_axis < 0 {
                                __raw_axis + (4usize as i64)
                            } else {
                                __raw_axis
                            };
                            assert!(
                                __dim_signed >= 0 && (__dim_signed as usize) < 4usize,
                                "Pad: axis {} out of range for rank {}", __raw_axis, 4usize,
                            );
                            let __dim = __dim_signed as usize;
                            assert!(
                                ! __seen[__dim], "Pad: duplicate axis {} (normalized to dim {})",
                                __raw_axis, __dim,
                            );
                            __seen[__dim] = true;
                            __pads[__dim] = (
                                usize::try_from(__raw_pads[__i])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw_pads[__i],
                                            __i
                                        )
                                    }),
                                usize::try_from(__raw_pads[__n + __i])
                                    .unwrap_or_else(|_| {
                                        panic!(
                                            "Pad: negative pad value {} at index {}", __raw_pads[__n +
                                            __i], __n + __i
                                        )
                                    }),
                            );
                        }
                        __pads
                    },
                    burn::tensor::ops::PadMode::Constant(0f32),
                );
            output
        }
        "#);
    }

    #[test]
    fn test_pad_4d_all_dimensions() {
        let config = PadConfig {
            pads: PadInput::Static(vec![(1, 2), (0, 0), (3, 4), (5, 6)]),
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = input
                .pad(
                    [(1usize, 2usize), (0usize, 0usize), (3usize, 4usize), (5usize, 6usize)],
                    burn::tensor::ops::PadMode::Constant(0f32),
                );
            output
        }
        ");
    }
}
