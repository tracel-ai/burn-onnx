use super::prelude::*;
use crate::burn::TensorKind;

impl NodeCodegen for onnx_ir::one_hot::OneHotNode {
    fn inputs(&self) -> &[Argument] {
        // Only the first input (indices) is a dynamic tensor
        // depth and values are either static or runtime inputs
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // Runtime depth/values are bound to `__onehot_*` locals inside a
        // prelude block so (1) the main one_hot_fill call stays readable
        // and (2) multiple OneHot nodes in the same forward() can't
        // collide on the temporary names.
        let mut prelude = TokenStream::new();

        let depth_expr = match &self.config.depth {
            onnx_ir::one_hot::OneHotDepthInput::Static(d) => quote! { #d },
            onnx_ir::one_hot::OneHotDepthInput::Runtime(r) => {
                let arg = &self.inputs[r.input_index];
                match &arg.ty {
                    ArgType::ScalarNative(_) => {
                        let ident = arg_to_ident(arg);
                        // Clamp with `max(0)` before the `as usize` cast.
                        // A negative runtime depth is out-of-spec for ONNX
                        // OneHot, but models in the wild occasionally emit
                        // broken graphs; letting a negative i64 wrap to a
                        // huge usize causes OOM or a cryptic deep-burn
                        // panic, so we clamp invalid negative depths to 0.
                        // This silently produces a zero-class one_hot result
                        // for broken models; adding observability
                        // (`log::warn!` or `debug_assert!`) is tracked in
                        // tracel-ai/burn-onnx#328.
                        prelude.extend(quote! {
                            let __onehot_depth: usize = (#ident as i64).max(0) as usize;
                        });
                    }
                    ArgType::ScalarTensor(_) | ArgType::Tensor(_) => {
                        let tensor = scope.arg(arg);
                        prelude.extend(quote! {
                            let __onehot_depth: usize = {
                                let __data = #tensor.to_data().convert::<i64>();
                                __data.as_slice::<i64>().unwrap()[0].max(0) as usize
                            };
                        });
                    }
                    other => {
                        panic!("OneHot depth must be a scalar or rank-1 tensor, got {other:?}")
                    }
                }
                quote! { __onehot_depth }
            }
        };

        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let num_classes = depth_expr;

        let axis = self.config.axis;

        // Determine input and output tensor kinds
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input_kind = match &input_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input"),
        };

        let output_kind = match &output_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor output"),
        };

        let output_dtype = output_arg.ty.elem_type();
        let output_dtype_tokens = output_dtype.to_tokens();

        // Runtime values widen through an intermediate scalar because Burn's
        // `one_hot_fill` pins `on_value`/`off_value` to `f32`. For a float
        // output, f32 matches the downstream dtype. For an int output, f32
        // rounds magnitudes above 2^24, so for the runtime+int case we
        // take a different path: call `one_hot_fill(1.0, 0.0)` to produce
        // a 0/1 mask, cast to a wide integer (i64 for signed output, u64
        // for unsigned output), scale via `mul_scalar`/`add_scalar` using
        // wrapping arithmetic, then narrow to the ONNX-specified output
        // dtype. `wrapping_sub` is safe here because the mask is always
        // 0 or 1:
        //     mask=0: 0 * (on - off) + off = off
        //     mask=1: (on - off) + off = on  (wrapping math cancels)
        // Picking `U64` for unsigned outputs preserves the full u64 range;
        // routing through `i64` would wrap values above i64::MAX.
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum RuntimeValuesMode {
            /// Output is floating-point; narrow values through f32.
            Float,
            /// Output is a signed integer; scale in i64 via wrapping math.
            SignedInt,
            /// Output is an unsigned integer; scale in u64 via wrapping math.
            UnsignedInt,
        }
        let runtime_values_mode = match output_kind {
            TensorKind::Int if output_dtype.is_uint() => RuntimeValuesMode::UnsignedInt,
            TensorKind::Int => RuntimeValuesMode::SignedInt,
            _ => RuntimeValuesMode::Float,
        };

        let (on_value, off_value) = match &self.config.values {
            onnx_ir::one_hot::OneHotValuesInput::Static(v) => {
                let off = v[0];
                let on = v[1];
                (quote! { #on }, quote! { #off })
            }
            onnx_ir::one_hot::OneHotValuesInput::Runtime(r) => {
                let arg = &self.inputs[r.input_index];
                let tensor = scope.arg(arg);
                match runtime_values_mode {
                    RuntimeValuesMode::UnsignedInt => {
                        // values layout: [off_value, on_value]. Read as
                        // u64 so the full u64 range survives.
                        prelude.extend(quote! {
                            let (__onehot_off_u, __onehot_on_u): (u64, u64) = {
                                let __data = #tensor.to_data().convert::<u64>();
                                let __slice = __data.as_slice::<u64>().unwrap();
                                (__slice[0], __slice[1])
                            };
                        });
                        // Use a 0/1 mask; the scale is applied later.
                        (quote! { 1f32 }, quote! { 0f32 })
                    }
                    RuntimeValuesMode::SignedInt => {
                        // values layout: [off_value, on_value]. Read as
                        // i64 so the full int64 range survives.
                        prelude.extend(quote! {
                            let (__onehot_off_i, __onehot_on_i): (i64, i64) = {
                                let __data = #tensor.to_data().convert::<i64>();
                                let __slice = __data.as_slice::<i64>().unwrap();
                                (__slice[0], __slice[1])
                            };
                        });
                        // Use a 0/1 mask; the scale is applied later.
                        (quote! { 1f32 }, quote! { 0f32 })
                    }
                    RuntimeValuesMode::Float => {
                        prelude.extend(quote! {
                            let (__onehot_off, __onehot_on): (f32, f32) = {
                                let __data = #tensor.to_data().convert::<f32>();
                                let __slice = __data.as_slice::<f32>().unwrap();
                                (__slice[0], __slice[1])
                            };
                        });
                        (quote! { __onehot_on }, quote! { __onehot_off })
                    }
                }
            }
        };

        // Build the `one_hot_fill` call as a trailing expression (no
        // `let #output = ...;`). Wrapping the prelude + expression inside
        // a single block scopes the `__onehot_*` temporaries so multiple
        // OneHot nodes in the same forward() don't collide.
        //
        // `one_hot_fill` returns a tensor whose element dtype comes from the
        // backend's default Int/Float element, not from the input tensor's
        // runtime dtype. Always cast to the ONNX-specified output dtype so
        // the generated code doesn't leak the backend default (CLAUDE.md).
        //
        // For the runtime-values int paths we call `one_hot_fill(1.0, 0.0)`
        // to get a 0/1 mask and scale via integer scalar math:
        //     result = mask * (on - off) + off
        // The backend default IntElem can be narrower than i64 (burn-flex
        // defaults to i32), so we force an explicit `.cast(I64|U64)` before
        // `mul_scalar`/`add_scalar`: the scale math then runs in a type
        // wide enough to represent every value the output dtype can hold,
        // and the final narrowing cast back to the ONNX output dtype
        // happens in one place. `wrapping_sub` is used so the unsigned
        // case computes correctly (mask=0 -> off; mask=1 -> on, with the
        // wrap canceling out).
        let runtime_values_is_runtime = matches!(
            self.config.values,
            onnx_ir::one_hot::OneHotValuesInput::Runtime(_)
        );
        let maybe_scale = if runtime_values_is_runtime {
            match runtime_values_mode {
                RuntimeValuesMode::UnsignedInt => quote! {
                    .cast(burn::tensor::DType::U64)
                        .mul_scalar(__onehot_on_u.wrapping_sub(__onehot_off_u))
                        .add_scalar(__onehot_off_u)
                },
                RuntimeValuesMode::SignedInt => quote! {
                    .cast(burn::tensor::DType::I64)
                        .mul_scalar(__onehot_on_i.wrapping_sub(__onehot_off_i))
                        .add_scalar(__onehot_off_i)
                },
                RuntimeValuesMode::Float => TokenStream::new(),
            }
        } else {
            TokenStream::new()
        };
        let body: TokenStream = match (input_kind, output_kind) {
            (TensorKind::Int, TensorKind::Int) => {
                quote! {
                    #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis) #maybe_scale .cast(#output_dtype_tokens)
                }
            }
            (TensorKind::Float, TensorKind::Float) => {
                quote! {
                    #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).cast(#output_dtype_tokens)
                }
            }
            (TensorKind::Int, TensorKind::Float) => {
                quote! {
                    #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).float().cast(#output_dtype_tokens)
                }
            }
            (TensorKind::Float, TensorKind::Int) => {
                quote! {
                    #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).int() #maybe_scale .cast(#output_dtype_tokens)
                }
            }
            (TensorKind::Int, TensorKind::Bool) | (TensorKind::Float, TensorKind::Bool) => {
                quote! {
                    #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).bool()
                }
            }
            (TensorKind::Bool, _) => panic!("Input should be numeric"),
        };

        if prelude.is_empty() {
            // Preserve the legacy flat form for the common static case so
            // snapshots stay readable and we don't introduce a redundant
            // block wrapper.
            quote! {
                let #output = #body;
            }
        } else {
            quote! {
                let #output = {
                    #prelude
                    #body
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::one_hot::{OneHotConfig, OneHotDepthInput, OneHotNodeBuilder, OneHotValuesInput};

    #[test]
    fn test_one_hot() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(10),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot1")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2> {
            let output = indices
                .one_hot_fill(10usize, 1f32, 0f32, -1i64)
                .float()
                .cast(burn::tensor::DType::F32);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_int_to_int() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot2")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2, Int> {
            let output = indices
                .one_hot_fill(5usize, 1f32, 0f32, -1i64)
                .cast(burn::tensor::DType::I32);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_float() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot3")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = indices
                .one_hot_fill(5usize, 1f32, 0f32, 0i64)
                .cast(burn::tensor::DType::F32);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_int() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot4")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2, Int> {
            let output = indices
                .one_hot_fill(5usize, 1f32, 0f32, 0i64)
                .int()
                .cast(burn::tensor::DType::I32);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_int_to_bool() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot5")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2, Bool> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, -1i64).bool();
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_runtime_depth_and_values() {
        // Opset 11+ gives OneHot all three inputs as tensors, which lands
        // here as Runtime depth and Runtime values.
        //
        // Depth dtype here intentionally mirrors the real ONNX backend
        // test `test_onehot_negative_indices`, which declares depth as
        // ONNX FLOAT (type 1) — the OneHot spec's T2 constraint allows any
        // numeric scalar, not just integers, and this path must accept
        // whatever the upstream model uses. See
        // `test_one_hot_runtime_depth_int64` below for the i64 variant.
        let config = OneHotConfig::new(
            OneHotDepthInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("depth".to_string(), 1)),
            OneHotValuesInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("values".to_string(), 2)),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot_rt")
            .input_tensor("indices", 1, DType::I64)
            .input_scalar_tensor("depth", DType::F32)
            .input_tensor("values", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            indices: Tensor<B, 1, Int>,
            depth: Tensor<B, 1>,
            values: Tensor<B, 1>,
        ) -> Tensor<B, 2> {
            let output = {
                let __onehot_depth: usize = {
                    let __data = depth.to_data().convert::<i64>();
                    __data.as_slice::<i64>().unwrap()[0].max(0) as usize
                };
                let (__onehot_off, __onehot_on): (f32, f32) = {
                    let __data = values.to_data().convert::<f32>();
                    let __slice = __data.as_slice::<f32>().unwrap();
                    (__slice[0], __slice[1])
                };
                indices
                    .one_hot_fill(__onehot_depth, __onehot_on, __onehot_off, -1i64)
                    .float()
                    .cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_runtime_depth_int64() {
        // Companion to test_one_hot_runtime_depth_and_values: some ONNX
        // models (and the spec's T2 constraint allows it) pass depth as
        // an i64 tensor rather than f32. The generated prelude should
        // produce identical code thanks to `to_data().convert::<i64>()`
        // being dtype-agnostic.
        let config = OneHotConfig::new(
            OneHotDepthInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("depth".to_string(), 1)),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot_rt_i64")
            .input_tensor("indices", 1, DType::I64)
            .input_scalar_tensor("depth", DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            indices: Tensor<B, 1, Int>,
            depth: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2> {
            let output = {
                let __onehot_depth: usize = {
                    let __data = depth.to_data().convert::<i64>();
                    __data.as_slice::<i64>().unwrap()[0].max(0) as usize
                };
                indices
                    .one_hot_fill(__onehot_depth, 1f32, 0f32, -1i64)
                    .float()
                    .cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_runtime_values_int_output() {
        // Runtime values fed into an int-typed output must preserve full
        // int64 precision. The old code path narrowed `values` through f32,
        // which silently rounded magnitudes above 2^24. The fix reads the
        // two values as i64, calls `one_hot_fill(1.0, 0.0)` to get a 0/1
        // mask, explicitly casts that mask to I64 (burn-flex's default
        // IntElem is i32, so an implicit `.int()` would still lose bits),
        // then scales via i64 scalar math before the final narrowing
        // cast to the ONNX-specified output dtype.
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("values".to_string(), 1)),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot_rt_int")
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("values", 1, DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            indices: Tensor<B, 1, Int>,
            values: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = {
                let (__onehot_off_i, __onehot_on_i): (i64, i64) = {
                    let __data = values.to_data().convert::<i64>();
                    let __slice = __data.as_slice::<i64>().unwrap();
                    (__slice[0], __slice[1])
                };
                indices
                    .one_hot_fill(5usize, 1f32, 0f32, -1i64)
                    .cast(burn::tensor::DType::I64)
                    .mul_scalar(__onehot_on_i.wrapping_sub(__onehot_off_i))
                    .add_scalar(__onehot_off_i)
                    .cast(burn::tensor::DType::I64)
            };
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_runtime_values_float_indices_int_output() {
        // Companion to test_one_hot_runtime_values_int_output but with
        // float indices, exercising the `(Float, Int)` match arm that
        // chains `.int()` before the int64 scale. The `.int()` narrows to
        // the backend default IntElem (i32 on burn-flex), and the
        // subsequent explicit `.cast(I64)` widens back to i64 so the
        // scale math stays precise regardless of backend.
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("values".to_string(), 1)),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot_rt_float_to_int")
            .input_tensor("indices", 1, DType::F32)
            .input_tensor("values", 1, DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            indices: Tensor<B, 1>,
            values: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = {
                let (__onehot_off_i, __onehot_on_i): (i64, i64) = {
                    let __data = values.to_data().convert::<i64>();
                    let __slice = __data.as_slice::<i64>().unwrap();
                    (__slice[0], __slice[1])
                };
                indices
                    .one_hot_fill(5usize, 1f32, 0f32, -1i64)
                    .int()
                    .cast(burn::tensor::DType::I64)
                    .mul_scalar(__onehot_on_i.wrapping_sub(__onehot_off_i))
                    .add_scalar(__onehot_off_i)
                    .cast(burn::tensor::DType::I64)
            };
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_runtime_values_uint_output() {
        // Runtime values with an unsigned output dtype (U64) must read
        // through u64 and scale in U64 so the full u64 range survives.
        // Routing through i64 would wrap any on/off value above i64::MAX
        // into a large negative intermediate, silently producing wrong
        // results after the final cast. `wrapping_sub` is safe because the
        // mask is 0 or 1: mask=0 -> 0 + off = off, mask=1 -> (on - off)
        // + off = on under wrapping arithmetic.
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Runtime(onnx_ir::ir::RuntimeInputRef::new("values".to_string(), 1)),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot_rt_uint")
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("values", 1, DType::U64)
            .output_tensor("output", 2, DType::U64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            indices: Tensor<B, 1, Int>,
            values: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = {
                let (__onehot_off_u, __onehot_on_u): (u64, u64) = {
                    let __data = values.to_data().convert::<u64>();
                    let __slice = __data.as_slice::<u64>().unwrap();
                    (__slice[0], __slice[1])
                };
                indices
                    .one_hot_fill(5usize, 1f32, 0f32, -1i64)
                    .cast(burn::tensor::DType::U64)
                    .mul_scalar(__onehot_on_u.wrapping_sub(__onehot_off_u))
                    .add_scalar(__onehot_off_u)
                    .cast(burn::tensor::DType::U64)
            };
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_bool() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot6")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2, Bool> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, 0i64).bool();
            output
        }
        ");
    }
}
