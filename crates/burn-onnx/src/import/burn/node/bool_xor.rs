use super::prelude::*;

impl NodeCodegen for onnx_ir::node::xor::XorNode {
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
                // XOR is implemented as not_equal for boolean tensors
                quote! { #lhs_bc.not_equal(#rhs_bc) }
            }
            (ArgType::ScalarNative(_), rhs_ty) if rhs_ty.is_on_device() => quote! {
                if #lhs { #rhs.bool_not() } else { #rhs }
            },
            (lhs_ty, ArgType::ScalarNative(_)) if lhs_ty.is_on_device() => quote! {
                if #rhs { #lhs.bool_not() } else { #lhs }
            },
            (ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => {
                quote! { #lhs ^ #rhs }
            }
            (ArgType::Shape(lhs_len), ArgType::Shape(rhs_len)) => {
                broadcast_helpers::shape_binary_elementwise(
                    quote! { #lhs },
                    *lhs_len,
                    quote! { #rhs },
                    *rhs_len,
                    |a, b| quote! { if (#a != 0) ^ (#b != 0) { 1i64 } else { 0i64 } },
                )
            }
            _ => panic!("Xor operation requires tensor, scalar, or shape inputs"),
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
    use onnx_ir::node::xor::XorNodeBuilder;

    #[test]
    fn test_xor_scalar_tensor_forward() {
        let node = XorNodeBuilder::new("xor1")
            .input_scalar("lhs", DType::Bool(BoolStore::Native))
            .input_tensor("rhs", 4, DType::Bool(BoolStore::Native))
            .output_tensor("output", 4, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: bool, rhs: Tensor<4, Bool>) -> Tensor<4, Bool> {
            let output = if lhs { rhs.bool_not() } else { rhs };
            output
        }
        ");
    }

    #[test]
    fn test_xor_tensor_scalar_forward() {
        let node = XorNodeBuilder::new("xor1")
            .input_tensor("lhs", 4, DType::Bool(BoolStore::Native))
            .input_scalar("rhs", DType::Bool(BoolStore::Native))
            .output_tensor("output", 4, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<4, Bool>, rhs: bool) -> Tensor<4, Bool> {
            let output = if rhs { lhs.bool_not() } else { lhs };
            output
        }
        ");
    }

    #[test]
    fn test_xor_forward() {
        let node = XorNodeBuilder::new("xor1")
            .input_tensor("lhs", 2, DType::Bool(BoolStore::Native))
            .input_tensor("rhs", 2, DType::Bool(BoolStore::Native))
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<2, Bool>, rhs: Tensor<2, Bool>) -> Tensor<2, Bool> {
            let output = lhs.not_equal(rhs);
            output
        }
        ");
    }

    // --- Shape + Shape ---

    #[test]
    fn test_shape_shape() {
        let node = XorNodeBuilder::new("xor1")
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
                    __result[__i] = if (__lhs[__i] != 0) ^ (__rhs[__i] != 0) { 1i64 } else { 0i64 };
                }
                __result
            };
            output
        }
        ");
    }

    #[test]
    fn test_shape_shape_broadcast_lhs() {
        let node = XorNodeBuilder::new("xor1")
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
                    __result[__i] = if (__lhs[0] != 0) ^ (__rhs[__i] != 0) { 1i64 } else { 0i64 };
                }
                __result
            };
            output
        }
        ");
    }

    #[test]
    fn test_shape_shape_broadcast_rhs() {
        let node = XorNodeBuilder::new("xor1")
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
                    __result[__i] = if (__lhs[__i] != 0) ^ (__rhs[0] != 0) { 1i64 } else { 0i64 };
                }
                __result
            };
            output
        }
        ");
    }
}
