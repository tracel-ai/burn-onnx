use super::prelude::*;

impl NodeCodegen for onnx_ir::node::and::AndNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs = scope.arg(lhs_arg);
        let rhs = scope.arg(rhs_arg);

        let function = match (&lhs_arg.ty, &rhs_arg.ty) {
            (lhs_ty, rhs_ty) if lhs_ty.is_on_device() && rhs_ty.is_on_device() => {
                let lhs_rank = lhs_ty.rank();
                let rhs_rank = rhs_ty.rank();
                let lhs_bc =
                    broadcast_helpers::leading_broadcast(quote! { #lhs }, lhs_rank, rhs_rank);
                let rhs_bc =
                    broadcast_helpers::leading_broadcast(quote! { #rhs }, rhs_rank, lhs_rank);
                quote! { #lhs_bc.bool_and(#rhs_bc) }
            }
            (ArgType::ScalarNative(_), rhs_ty) if rhs_ty.is_on_device() => {
                let rank = rhs_ty.rank();
                quote! {
                    if #lhs { #rhs } else { Tensor::<#rank, Int>::zeros(#rhs.shape(), &self.device).bool() }
                }
            }
            (lhs_ty, ArgType::ScalarNative(_)) if lhs_ty.is_on_device() => {
                let rank = lhs_ty.rank();
                quote! {
                    if #rhs { #lhs } else { Tensor::<#rank, Int>::zeros(#lhs.shape(), &self.device).bool() }
                }
            }
            (ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => {
                quote! { #lhs && #rhs }
            }
            (ArgType::Shape(lhs_len), ArgType::Shape(rhs_len)) => {
                broadcast_helpers::shape_binary_elementwise(
                    quote! { #lhs },
                    *lhs_len,
                    quote! { #rhs },
                    *rhs_len,
                    |a, b| quote! { if #a != 0 && #b != 0 { 1i64 } else { 0i64 } },
                )
            }
            // `And` on a Shape happens when a prior Shape-returning op
            // (e.g. elementwise comparison of two Shapes in comparison.rs,
            // which emits a Shape of 0/1 i64s) feeds into And. Convert the
            // Shape to a 1D Int tensor and compare with 0 to get ONNX's
            // "non-zero is truthy" Bool semantics, then reuse the on-device
            // bool_and path.
            (ArgType::Shape(_), rhs_ty) if rhs_ty.is_on_device() => {
                quote! {
                    Tensor::<1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::from(&#lhs as &[i64]),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .not_equal_elem(0i64)
                    .bool_and(#rhs)
                }
            }
            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_on_device() => {
                quote! {
                    #lhs.bool_and(
                        Tensor::<1, burn::tensor::Int>::from_data(
                            burn::tensor::TensorData::from(&#rhs as &[i64]),
                            (&self.device, burn::tensor::DType::I64),
                        )
                        .not_equal_elem(0i64)
                    )
                }
            }
            _ => panic!(
                "And operation: unsupported input types: lhs={:?}, rhs={:?}",
                lhs_arg.ty, rhs_arg.ty
            ),
        };

        quote! {
            let #output = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::node::and::AndNodeBuilder;

    #[test]
    fn test_and_scalar_tensor_forward() {
        let node = AndNodeBuilder::new("and1")
            .input_scalar("lhs", DType::Bool(BoolStore::Native))
            .input_tensor("rhs", 4, DType::Bool(BoolStore::Native))
            .output_tensor("output", 4, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: bool, rhs: Tensor<4, Bool>) -> Tensor<4, Bool> {
            let output = if lhs {
                rhs
            } else {
                Tensor::<4usize, Int>::zeros(rhs.shape(), &self.device).bool()
            };
            output
        }
        ");
    }

    #[test]
    fn test_and_tensor_scalar_forward() {
        let node = AndNodeBuilder::new("and1")
            .input_tensor("lhs", 4, DType::Bool(BoolStore::Native))
            .input_scalar("rhs", DType::Bool(BoolStore::Native))
            .output_tensor("output", 4, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<4, Bool>, rhs: bool) -> Tensor<4, Bool> {
            let output = if rhs {
                lhs
            } else {
                Tensor::<4usize, Int>::zeros(lhs.shape(), &self.device).bool()
            };
            output
        }
        ");
    }

    #[test]
    fn test_and_forward() {
        let node = AndNodeBuilder::new("and1")
            .input_tensor("lhs", 2, DType::Bool(BoolStore::Native))
            .input_tensor("rhs", 2, DType::Bool(BoolStore::Native))
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<2, Bool>, rhs: Tensor<2, Bool>) -> Tensor<2, Bool> {
            let output = lhs.bool_and(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_and_shape_bool_tensor() {
        let node = AndNodeBuilder::new("and1")
            .input_shape("lhs", 1)
            .input_tensor("rhs", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: [i64; 1], rhs: Tensor<1, Bool>) -> Tensor<1, Bool> {
            let output = Tensor::<
                1,
                burn::tensor::Int,
            >::from_data(
                    burn::tensor::TensorData::from(&lhs as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                )
                .not_equal_elem(0i64)
                .bool_and(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_and_bool_tensor_shape() {
        let node = AndNodeBuilder::new("and1")
            .input_tensor("lhs", 1, DType::Bool(BoolStore::Native))
            .input_shape("rhs", 1)
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<1, Bool>, rhs: [i64; 1]) -> Tensor<1, Bool> {
            let output = lhs
                .bool_and(
                    Tensor::<
                        1,
                        burn::tensor::Int,
                    >::from_data(
                            burn::tensor::TensorData::from(&rhs as &[i64]),
                            (&self.device, burn::tensor::DType::I64),
                        )
                        .not_equal_elem(0i64),
                );
            output
        }
        ");
    }

    // --- Shape + Shape ---

    #[test]
    fn test_shape_shape() {
        let node = AndNodeBuilder::new("and1")
            .input_shape("lhs", 4)
            .input_shape("rhs", 4)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: [i64; 4]) -> [i64; 4] {
            let output = {
                let __lhs = lhs;
                let __rhs = rhs;
                let mut __result = [0i64; 4usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..4usize {
                    __result[__i] = if __lhs[__i] != 0 && __rhs[__i] != 0 { 1i64 } else { 0i64 };
                }
                __result
            };
            output
        }
        ");
    }

    #[test]
    fn test_shape_shape_broadcast_lhs() {
        let node = AndNodeBuilder::new("and1")
            .input_shape("lhs", 1)
            .input_shape("rhs", 4)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 1], rhs: [i64; 4]) -> [i64; 4] {
            let output = {
                let __lhs = lhs;
                let __rhs = rhs;
                let mut __result = [0i64; 4usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..4usize {
                    __result[__i] = if __lhs[0] != 0 && __rhs[__i] != 0 { 1i64 } else { 0i64 };
                }
                __result
            };
            output
        }
        ");
    }

    #[test]
    fn test_shape_shape_broadcast_rhs() {
        let node = AndNodeBuilder::new("and1")
            .input_shape("lhs", 4)
            .input_shape("rhs", 1)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: [i64; 1]) -> [i64; 4] {
            let output = {
                let __lhs = lhs;
                let __rhs = rhs;
                let mut __result = [0i64; 4usize];
                #[allow(clippy::needless_range_loop)]
                for __i in 0..4usize {
                    __result[__i] = if __lhs[__i] != 0 && __rhs[0] != 0 { 1i64 } else { 0i64 };
                }
                __result
            };
            output
        }
        ");
    }
}
