use super::prelude::*;

impl NodeCodegen for onnx_ir::modulo::ModNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs_arg = &self.inputs[0];
        let rhs_arg = &self.inputs[1];

        match (&lhs_arg.ty, &rhs_arg.ty) {
            (lhs_ty, rhs_ty) if lhs_ty.is_on_device() && rhs_ty.is_on_device() => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);

                let lhs_rank = lhs_ty.rank();
                let rhs_rank = rhs_ty.rank();

                let lhs_bc =
                    broadcast_helpers::leading_broadcast(quote! { #lhs }, lhs_rank, rhs_rank);
                let rhs_bc =
                    broadcast_helpers::leading_broadcast(quote! { #rhs }, rhs_rank, lhs_rank);

                let expr = if self.config.fmod {
                    quote! { #lhs_bc.fmod(#rhs_bc) }
                } else {
                    // Burn's remainder does not broadcast internally
                    let output_rank = lhs_rank.max(rhs_rank);
                    broadcast_helpers::broadcast_binary_op(
                        lhs_bc,
                        rhs_bc,
                        output_rank,
                        quote! { remainder },
                    )
                };
                quote! {
                    let #output = #expr;
                }
            }
            (lhs_ty, ArgType::ScalarNative(_)) if lhs_ty.is_on_device() => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);

                let mod_op = if self.config.fmod {
                    quote! { fmod_scalar }
                } else {
                    quote! { remainder_scalar }
                };

                quote! {
                    let #output = #lhs.#mod_op(#rhs);
                }
            }
            (ArgType::ScalarNative(lhs_dtype), ArgType::ScalarNative(_)) => {
                let lhs = arg_to_ident(lhs_arg);
                let rhs = arg_to_ident(rhs_arg);
                // ONNX Mod with fmod=0 (default) is Python-style: the result
                // has the sign of the divisor. Rust's `%` is C-style: the
                // result has the sign of the dividend. They coincide on
                // non-negative integers and on floats, but diverge on signed
                // integers with a negative operand. Use `rem_euclid` for the
                // signed-integer fmod=0 case to get Python semantics. For
                // floats and for fmod=1 (C-style fmod), `%` is already
                // correct. Unsigned ints can never produce a negative result
                // so `%` is also correct there.
                let expr = if !self.config.fmod && lhs_dtype.is_int() {
                    quote! { #lhs.rem_euclid(#rhs) }
                } else {
                    quote! { #lhs % #rhs }
                };
                quote! {
                    let #output = #expr;
                }
            }
            (ArgType::ScalarNative(dtype), rhs_ty) if rhs_ty.is_on_device() => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);
                let dtype_tokens = dtype.to_tokens();
                let rhs_rank = rhs_ty.rank();

                let (tensor_type, cast_as) = if dtype.is_float() {
                    (quote! { Tensor::<B, 1> }, quote! { f64 })
                } else {
                    (quote! { Tensor::<B, 1, burn::tensor::Int> }, quote! { i64 })
                };

                let lhs_tensor = if rhs_rank > 1 {
                    let dims: Vec<isize> = (0..rhs_rank - 1).map(|i| i as isize).collect();
                    quote! {
                        #tensor_type::from_data(
                            burn::tensor::TensorData::from([#lhs as #cast_as]),
                            (&self.device, #dtype_tokens)
                        ).unsqueeze_dims(&[#(#dims),*])
                    }
                } else {
                    quote! {
                        #tensor_type::from_data(
                            burn::tensor::TensorData::from([#lhs as #cast_as]),
                            (&self.device, #dtype_tokens)
                        )
                    }
                };

                let expr = if self.config.fmod {
                    quote! { #lhs_tensor.fmod(#rhs) }
                } else {
                    broadcast_helpers::broadcast_binary_op(
                        lhs_tensor,
                        quote! { #rhs },
                        rhs_rank,
                        quote! { remainder },
                    )
                };
                quote! {
                    let #output = #expr;
                }
            }
            // Shape values are `[i64; N]` arrays representing tensor dimensions.
            // In practice they come from the ONNX `Shape` op (always non-negative)
            // or from Shape arithmetic like sub.rs, which uses `saturating_sub` to
            // clamp at 0. For non-negative integers Rust's `%`, ONNX fmod, and ONNX
            // remainder all coincide, so a plain elementwise `%` covers both
            // config.fmod variants here. If a future producer ever lets negative
            // Shape values reach this arm, the `fmod=0` path would need to switch
            // to `rem_euclid` to match ONNX semantics.
            (ArgType::Shape(_), ArgType::Shape(_)) => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);
                quote! {
                    let #output = {
                        let mut result = #lhs;
                        for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                            *result_item %= *rhs_item;
                        }
                        result
                    };
                }
            }
            (ArgType::Shape(_), rhs_ty) if rhs_ty.is_scalar() => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);
                let scalar_expr = if rhs_ty.is_scalar_tensor() {
                    on_device_to_native(rhs.clone(), &rhs_ty.elem_type())
                } else {
                    quote! { #rhs as i64 }
                };
                quote! {
                    let #output = {
                        let mut result = #lhs;
                        let __scalar = #scalar_expr;
                        for result_item in result.iter_mut() {
                            *result_item %= __scalar;
                        }
                        result
                    };
                }
            }
            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_scalar() => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);
                let scalar_expr = if lhs_ty.is_scalar_tensor() {
                    on_device_to_native(lhs.clone(), &lhs_ty.elem_type())
                } else {
                    quote! { #lhs as i64 }
                };
                quote! {
                    let #output = {
                        let mut result = #rhs;
                        let __scalar = #scalar_expr;
                        for result_item in result.iter_mut() {
                            *result_item = __scalar % *result_item;
                        }
                        result
                    };
                }
            }
            (ArgType::Shape(_), rhs_ty) if rhs_ty.is_on_device() => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);
                let dtype_tokens = rhs_ty.elem_type().to_tokens();
                let lhs_tensor = quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::from(&#lhs as &[i64]),
                        (&self.device, #dtype_tokens)
                    )
                };
                let expr = if self.config.fmod {
                    quote! { #lhs_tensor.fmod(#rhs) }
                } else {
                    quote! { #lhs_tensor.remainder(#rhs) }
                };
                quote! {
                    let #output = #expr;
                }
            }
            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_on_device() => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);
                let dtype_tokens = lhs_ty.elem_type().to_tokens();
                let rhs_tensor = quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::from(&#rhs as &[i64]),
                        (&self.device, #dtype_tokens)
                    )
                };
                let expr = if self.config.fmod {
                    quote! { #lhs.fmod(#rhs_tensor) }
                } else {
                    quote! { #lhs.remainder(#rhs_tensor) }
                };
                quote! {
                    let #output = #expr;
                }
            }
            _ => panic!(
                "Unsupported Mod input types: lhs={:?}, rhs={:?}",
                lhs_arg.ty, rhs_arg.ty
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::modulo::{ModConfig, ModNodeBuilder};

    // --- on_device + on_device (same rank) ---

    #[test]
    fn test_remainder_tensor_tensor_same_rank() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let __lhs = a;
                let __rhs = b;
                let __lhs_dims: [usize; 2usize] = __lhs.dims();
                let __rhs_dims: [usize; 2usize] = __rhs.dims();
                let mut __shape = [0i64; 2usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..2usize {
                    __shape[__i] = core::cmp::max(
                        __lhs_dims[__i] as i64,
                        __rhs_dims[__i] as i64,
                    );
                }
                __lhs.expand(__shape).remainder(__rhs.expand(__shape))
            };
            output
        }
        ");
    }

    #[test]
    fn test_fmod_tensor_tensor_same_rank() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.fmod(b);
            output
        }
        ");
    }

    // --- on_device + on_device (broadcast) ---

    #[test]
    fn test_remainder_broadcast_lhs_smaller() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let __lhs = (a).unsqueeze_dims(&[0isize]);
                let __rhs = b;
                let __lhs_dims: [usize; 3usize] = __lhs.dims();
                let __rhs_dims: [usize; 3usize] = __rhs.dims();
                let mut __shape = [0i64; 3usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..3usize {
                    __shape[__i] = core::cmp::max(
                        __lhs_dims[__i] as i64,
                        __rhs_dims[__i] as i64,
                    );
                }
                __lhs.expand(__shape).remainder(__rhs.expand(__shape))
            };
            output
        }
        ");
    }

    #[test]
    fn test_remainder_broadcast_rhs_smaller() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 3, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = {
                let __lhs = a;
                let __rhs = (b).unsqueeze_dims(&[0isize]);
                let __lhs_dims: [usize; 3usize] = __lhs.dims();
                let __rhs_dims: [usize; 3usize] = __rhs.dims();
                let mut __shape = [0i64; 3usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..3usize {
                    __shape[__i] = core::cmp::max(
                        __lhs_dims[__i] as i64,
                        __rhs_dims[__i] as i64,
                    );
                }
                __lhs.expand(__shape).remainder(__rhs.expand(__shape))
            };
            output
        }
        ");
    }

    #[test]
    fn test_fmod_broadcast_lhs_smaller() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = (a).unsqueeze_dims(&[0isize]).fmod(b);
            output
        }
        ");
    }

    #[test]
    fn test_fmod_broadcast_rhs_smaller() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 3, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = a.fmod((b).unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    // --- on_device + on_device (ScalarTensor broadcast) ---

    #[test]
    fn test_remainder_tensor_scalar_tensor() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 3, DType::F32)
            .input_scalar_tensor("b", DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 1>) -> Tensor<B, 3> {
            let output = {
                let __lhs = a;
                let __rhs = (b).unsqueeze_dims(&[0isize, 1isize]);
                let __lhs_dims: [usize; 3usize] = __lhs.dims();
                let __rhs_dims: [usize; 3usize] = __rhs.dims();
                let mut __shape = [0i64; 3usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..3usize {
                    __shape[__i] = core::cmp::max(
                        __lhs_dims[__i] as i64,
                        __rhs_dims[__i] as i64,
                    );
                }
                __lhs.expand(__shape).remainder(__rhs.expand(__shape))
            };
            output
        }
        ");
    }

    #[test]
    fn test_fmod_tensor_scalar_tensor() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 3, DType::F32)
            .input_scalar_tensor("b", DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 1>) -> Tensor<B, 3> {
            let output = a.fmod((b).unsqueeze_dims(&[0isize, 1isize]));
            output
        }
        ");
    }

    #[test]
    fn test_remainder_scalar_tensor_scalar_tensor() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar_tensor("a", DType::F32)
            .input_scalar_tensor("b", DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
            let output = {
                let __lhs = a;
                let __rhs = b;
                let __lhs_dims: [usize; 1usize] = __lhs.dims();
                let __rhs_dims: [usize; 1usize] = __rhs.dims();
                let mut __shape = [0i64; 1usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..1usize {
                    __shape[__i] = core::cmp::max(
                        __lhs_dims[__i] as i64,
                        __rhs_dims[__i] as i64,
                    );
                }
                __lhs.expand(__shape).remainder(__rhs.expand(__shape))
            };
            output
        }
        ");
    }

    // --- ScalarNative + ScalarNative ---

    #[test]
    fn test_remainder_scalar_native_scalar_native_signed_int() {
        // ONNX Mod fmod=0 on signed ints is Python-style: sign follows
        // divisor. Rust's `%` is C-style (sign follows dividend), so we
        // must lower to `rem_euclid` to match ONNX semantics on negative
        // inputs. Without this fix, Mod(-7, 3) would emit `a % b`, which
        // evaluates to -1 in Rust versus ONNX's 2.
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar("a", DType::I64)
            .input_scalar("b", DType::I64)
            .output_scalar("output", DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: i64, b: i64) -> i64 {
            let output = a.rem_euclid(b);
            output
        }
        ");
    }

    #[test]
    fn test_fmod_scalar_native_scalar_native_signed_int() {
        // fmod=1 is C-style (sign follows dividend), which matches Rust's
        // `%` directly. No `rem_euclid` call.
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar("a", DType::I64)
            .input_scalar("b", DType::I64)
            .output_scalar("output", DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: i64, b: i64) -> i64 {
            let output = a % b;
            output
        }
        ");
    }

    #[test]
    fn test_remainder_scalar_native_scalar_native_float() {
        // ONNX Mod with `fmod=0` on float operands is out of spec: the
        // ONNX Mod op requires `fmod=1` whenever the operands are
        // floating-point, so a well-formed graph never reaches this
        // branch with floats and `fmod=0`. We keep Rust's `%` here
        // because it matches C-style fmod (which is what `fmod=1` asks
        // for) and because the signed-int-only `rem_euclid` guard
        // naturally skips floats. This test just pins that the emitted
        // code is a plain `%` so any future refactor that conflates the
        // float and signed-int arms shows up as a snapshot diff.
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar("a", DType::F32)
            .input_scalar("b", DType::F32)
            .output_scalar("output", DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: f32, b: f32) -> f32 {
            let output = a % b;
            output
        }
        ");
    }

    // --- on_device + ScalarNative ---

    #[test]
    fn test_remainder_tensor_scalar_native() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 2, DType::F32)
            .input_scalar("b", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: f32) -> Tensor<B, 2> {
            let output = a.remainder_scalar(b);
            output
        }
        ");
    }

    #[test]
    fn test_fmod_tensor_scalar_native() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 2, DType::F32)
            .input_scalar("b", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: f32) -> Tensor<B, 2> {
            let output = a.fmod_scalar(b);
            output
        }
        ");
    }

    // --- ScalarNative + on_device ---

    #[test]
    fn test_remainder_scalar_native_tensor() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar("a", DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: f32, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let __lhs = Tensor::<
                    B,
                    1,
                >::from_data(
                        burn::tensor::TensorData::from([a as f64]),
                        (&self.device, burn::tensor::DType::F32),
                    )
                    .unsqueeze_dims(&[0isize]);
                let __rhs = b;
                let __lhs_dims: [usize; 2usize] = __lhs.dims();
                let __rhs_dims: [usize; 2usize] = __rhs.dims();
                let mut __shape = [0i64; 2usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..2usize {
                    __shape[__i] = core::cmp::max(
                        __lhs_dims[__i] as i64,
                        __rhs_dims[__i] as i64,
                    );
                }
                __lhs.expand(__shape).remainder(__rhs.expand(__shape))
            };
            output
        }
        ");
    }

    #[test]
    fn test_fmod_scalar_native_tensor() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar("a", DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, a: f32, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = Tensor::<
                B,
                1,
            >::from_data(
                    burn::tensor::TensorData::from([a as f64]),
                    (&self.device, burn::tensor::DType::F32),
                )
                .unsqueeze_dims(&[0isize])
                .fmod(b);
            output
        }
        ");
    }

    // --- Shape + Shape ---

    #[test]
    fn test_mod_shape_shape() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_shape("lhs", 3)
            .input_shape("rhs", 3)
            .output_shape("output", 3)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 3], rhs: [i64; 3]) -> [i64; 3] {
            let output = {
                let mut result = lhs;
                for (result_item, rhs_item) in result.iter_mut().zip(rhs.iter()) {
                    *result_item %= *rhs_item;
                }
                result
            };
            output
        }
        ");
    }

    // --- Shape + ScalarNative ---

    #[test]
    fn test_mod_shape_scalar_native() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_shape("lhs", 3)
            .input_scalar("rhs", DType::I64)
            .output_shape("output", 3)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 3], rhs: i64) -> [i64; 3] {
            let output = {
                let mut result = lhs;
                let __scalar = rhs as i64;
                for result_item in result.iter_mut() {
                    *result_item %= __scalar;
                }
                result
            };
            output
        }
        ");
    }

    // --- ScalarNative + Shape ---

    #[test]
    fn test_mod_scalar_native_shape() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar("lhs", DType::I64)
            .input_shape("rhs", 3)
            .output_shape("output", 3)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: i64, rhs: [i64; 3]) -> [i64; 3] {
            let output = {
                let mut result = rhs;
                let __scalar = lhs as i64;
                for result_item in result.iter_mut() {
                    *result_item = __scalar % *result_item;
                }
                result
            };
            output
        }
        ");
    }

    // --- Shape + on_device ---

    #[test]
    fn test_mod_shape_tensor() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_shape("lhs", 3)
            .input_tensor("rhs", 1, DType::I64)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 3], rhs: Tensor<B, 1, Int>) -> Tensor<B, 1, Int> {
            let output = Tensor::<
                B,
                1,
                burn::tensor::Int,
            >::from_data(
                    burn::tensor::TensorData::from(&lhs as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                )
                .remainder(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_fmod_shape_tensor() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_shape("lhs", 3)
            .input_tensor("rhs", 1, DType::I64)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 3], rhs: Tensor<B, 1, Int>) -> Tensor<B, 1, Int> {
            let output = Tensor::<
                B,
                1,
                burn::tensor::Int,
            >::from_data(
                    burn::tensor::TensorData::from(&lhs as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                )
                .fmod(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_mod_tensor_shape() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("lhs", 1, DType::I64)
            .input_shape("rhs", 3)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 1, Int>, rhs: [i64; 3]) -> Tensor<B, 1, Int> {
            let output = lhs
                .remainder(
                    Tensor::<
                        B,
                        1,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::from(&rhs as &[i64]),
                        (&self.device, burn::tensor::DType::I64),
                    ),
                );
            output
        }
        ");
    }

    #[test]
    fn test_fmod_tensor_shape() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("lhs", 1, DType::I64)
            .input_shape("rhs", 3)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 1, Int>, rhs: [i64; 3]) -> Tensor<B, 1, Int> {
            let output = lhs
                .fmod(
                    Tensor::<
                        B,
                        1,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::from(&rhs as &[i64]),
                        (&self.device, burn::tensor::DType::I64),
                    ),
                );
            output
        }
        ");
    }

    // `fmod` coverage for the Shape/Shape, Shape/ScalarNative, and
    // ScalarNative/Shape arms: these arms deliberately ignore config.fmod
    // because Shape values are non-negative in practice and Rust's `%`
    // coincides with both fmod and remainder on non-negative integers.
    // These tests pin that behavior so any accidental split of an arm into
    // fmod/remainder branches shows up as a snapshot diff.

    #[test]
    fn test_fmod_shape_shape() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_shape("lhs", 3)
            .input_shape("rhs", 3)
            .output_shape("output", 3)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 3], rhs: [i64; 3]) -> [i64; 3] {
            let output = {
                let mut result = lhs;
                for (result_item, rhs_item) in result.iter_mut().zip(rhs.iter()) {
                    *result_item %= *rhs_item;
                }
                result
            };
            output
        }
        ");
    }

    #[test]
    fn test_fmod_shape_scalar_native() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_shape("lhs", 3)
            .input_scalar("rhs", DType::I64)
            .output_shape("output", 3)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 3], rhs: i64) -> [i64; 3] {
            let output = {
                let mut result = lhs;
                let __scalar = rhs as i64;
                for result_item in result.iter_mut() {
                    *result_item %= __scalar;
                }
                result
            };
            output
        }
        ");
    }

    #[test]
    fn test_fmod_scalar_native_shape() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar("lhs", DType::I64)
            .input_shape("rhs", 3)
            .output_shape("output", 3)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: i64, rhs: [i64; 3]) -> [i64; 3] {
            let output = {
                let mut result = rhs;
                let __scalar = lhs as i64;
                for result_item in result.iter_mut() {
                    *result_item = __scalar % *result_item;
                }
                result
            };
            output
        }
        ");
    }

    // Shape + ScalarTensor variants exercise the `is_scalar_tensor()` arm
    // of `Shape op scalar` / `scalar op Shape`, which uses `on_device_to_native`
    // to pull the scalar out of a rank-1 tensor.

    #[test]
    fn test_mod_shape_scalar_tensor() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_shape("lhs", 3)
            .input_scalar_tensor("rhs", DType::I64)
            .output_shape("output", 3)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 3], rhs: Tensor<B, 1, Int>) -> [i64; 3] {
            let output = {
                let mut result = lhs;
                let __scalar = rhs.into_scalar().elem::<i64>();
                for result_item in result.iter_mut() {
                    *result_item %= __scalar;
                }
                result
            };
            output
        }
        ");
    }

    #[test]
    fn test_mod_scalar_tensor_shape() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_scalar_tensor("lhs", DType::I64)
            .input_shape("rhs", 3)
            .output_shape("output", 3)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 1, Int>, rhs: [i64; 3]) -> [i64; 3] {
            let output = {
                let mut result = rhs;
                let __scalar = lhs.into_scalar().elem::<i64>();
                for result_item in result.iter_mut() {
                    *result_item = __scalar % *result_item;
                }
                result
            };
            output
        }
        ");
    }
}
