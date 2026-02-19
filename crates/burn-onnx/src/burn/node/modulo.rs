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

                // Handle broadcasting if ranks differ
                if lhs_rank != rhs_rank {
                    let (smaller_tensor, larger_tensor, smaller_rank, larger_rank) =
                        if lhs_rank < rhs_rank {
                            (&lhs, &rhs, lhs_rank, rhs_rank)
                        } else {
                            (&rhs, &lhs, rhs_rank, lhs_rank)
                        };

                    // Calculate dimensions to unsqueeze
                    let rank_diff = larger_rank - smaller_rank;
                    let unsqueeze_dims = (0..rank_diff)
                        .map(|i| {
                            let i = i as isize;
                            quote! { #i }
                        })
                        .collect::<Vec<_>>();

                    let mod_op = if self.config.fmod {
                        quote! { fmod }
                    } else {
                        quote! { remainder }
                    };

                    if lhs_rank < rhs_rank {
                        quote! {
                            let #output = #smaller_tensor
                                .unsqueeze_dims(&[#(#unsqueeze_dims),*])
                                .#mod_op(#larger_tensor);
                        }
                    } else {
                        quote! {
                            let #output = #larger_tensor.#mod_op(#smaller_tensor.unsqueeze_dims(&[#(#unsqueeze_dims),*]));
                        }
                    }
                } else {
                    let mod_op = if self.config.fmod {
                        quote! { fmod }
                    } else {
                        quote! { remainder }
                    };
                    quote! {
                        let #output = #lhs.#mod_op(#rhs);
                    }
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
            (ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => {
                let lhs = arg_to_ident(lhs_arg);
                let rhs = arg_to_ident(rhs_arg);
                quote! {
                    let #output = #lhs % #rhs;
                }
            }
            (ArgType::ScalarNative(dtype), rhs_ty) if rhs_ty.is_on_device() => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);
                let dtype_tokens = dtype.to_tokens();
                let rhs_rank = rhs_ty.rank();
                let mod_op = if self.config.fmod {
                    quote! { fmod }
                } else {
                    quote! { remainder }
                };

                let (tensor_type, cast_as) = if dtype.is_float() {
                    (quote! { Tensor::<B, 1> }, quote! { f64 })
                } else {
                    (quote! { Tensor::<B, 1, burn::tensor::Int> }, quote! { i64 })
                };

                if rhs_rank > 1 {
                    let dims: Vec<isize> = (0..rhs_rank - 1).map(|i| i as isize).collect();
                    quote! {
                        let #output = #tensor_type::from_data_dtype(
                            burn::tensor::TensorData::from([#lhs as #cast_as]),
                            &*self.device,
                            #dtype_tokens
                        ).unsqueeze_dims(&[#(#dims),*]).#mod_op(#rhs);
                    }
                } else {
                    quote! {
                        let #output = #tensor_type::from_data_dtype(
                            burn::tensor::TensorData::from([#lhs as #cast_as]),
                            &*self.device,
                            #dtype_tokens
                        ).#mod_op(#rhs);
                    }
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
            let output = a.remainder(b);
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
            let output = a.unsqueeze_dims(&[0isize]).remainder(b);
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
            let output = a.remainder(b.unsqueeze_dims(&[0isize]));
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
            let output = a.unsqueeze_dims(&[0isize]).fmod(b);
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
            let output = a.fmod(b.unsqueeze_dims(&[0isize]));
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
            let output = a.remainder(b.unsqueeze_dims(&[0isize, 1isize]));
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
            let output = a.fmod(b.unsqueeze_dims(&[0isize, 1isize]));
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
            let output = a.remainder(b);
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
            let output = Tensor::<
                B,
                1,
            >::from_data_dtype(
                    burn::tensor::TensorData::from([a as f64]),
                    &*self.device,
                    burn::tensor::DType::F32,
                )
                .unsqueeze_dims(&[0isize])
                .remainder(b);
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
            >::from_data_dtype(
                    burn::tensor::TensorData::from([a as f64]),
                    &*self.device,
                    burn::tensor::DType::F32,
                )
                .unsqueeze_dims(&[0isize])
                .fmod(b);
            output
        }
        ");
    }
}
