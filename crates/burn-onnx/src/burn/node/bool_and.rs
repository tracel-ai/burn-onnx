use super::prelude::*;

impl NodeCodegen for onnx_ir::node::and::AndNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs = self.inputs.first().unwrap();
        let rhs = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs_value = scope.arg(lhs);

        let rhs_value = scope.arg(rhs);

        let function = match (&lhs.ty, &rhs.ty) {
            (lhs_ty, rhs_ty) if lhs_ty.is_on_device() && rhs_ty.is_on_device() => {
                let lhs_rank = lhs_ty.rank();
                let rhs_rank = rhs_ty.rank();
                let lhs_bc =
                    broadcast_helpers::leading_broadcast(quote! { #lhs_value }, lhs_rank, rhs_rank);
                let rhs_bc =
                    broadcast_helpers::leading_broadcast(quote! { #rhs_value }, rhs_rank, lhs_rank);
                quote! { #lhs_bc.bool_and(#rhs_bc) }
            }
            (ArgType::ScalarNative(_), rhs_ty) if rhs_ty.is_on_device() => {
                let rank = rhs_ty.rank();
                quote! {
                    if #lhs_value { #rhs_value } else { Tensor::<B, #rank, Int>::zeros(#rhs_value.shape(), &self.device).bool() }
                }
            }
            (lhs_ty, ArgType::ScalarNative(_)) if lhs_ty.is_on_device() => {
                let rank = lhs_ty.rank();
                quote! {
                    if #rhs_value { #lhs_value } else { Tensor::<B, #rank, Int>::zeros(#lhs_value.shape(), &self.device).bool() }
                }
            }
            (ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => {
                quote! { #lhs_value && #rhs_value }
            }
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs_value;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs_value.iter()) {
                        *result_item = if *result_item != 0 && *rhs_item != 0 { 1i64 } else { 0i64 };
                    }
                    result
                }
            },
            // `And` on a Shape happens when a prior Shape-returning op
            // (e.g. elementwise comparison of two Shapes in comparison.rs,
            // which emits a Shape of 0/1 i64s) feeds into And. Convert the
            // Shape to a 1D Int tensor and compare with 0 to get ONNX's
            // "non-zero is truthy" Bool semantics, then reuse the on-device
            // bool_and path.
            (ArgType::Shape(_), rhs_ty) if rhs_ty.is_on_device() => {
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::from(&#lhs_value as &[i64]),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .not_equal_elem(0i64)
                    .bool_and(#rhs_value)
                }
            }
            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_on_device() => {
                quote! {
                    #lhs_value.bool_and(
                        Tensor::<B, 1, burn::tensor::Int>::from_data(
                            burn::tensor::TensorData::from(&#rhs_value as &[i64]),
                            (&self.device, burn::tensor::DType::I64),
                        )
                        .not_equal_elem(0i64)
                    )
                }
            }
            _ => panic!(
                "And operation: unsupported input types: lhs={:?}, rhs={:?}",
                lhs.ty, rhs.ty
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
        pub fn forward(&self, lhs: bool, rhs: Tensor<B, 4, Bool>) -> Tensor<B, 4, Bool> {
            let output = if lhs {
                rhs
            } else {
                Tensor::<B, 4usize, Int>::zeros(rhs.shape(), &self.device).bool()
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
        pub fn forward(&self, lhs: Tensor<B, 4, Bool>, rhs: bool) -> Tensor<B, 4, Bool> {
            let output = if rhs {
                lhs
            } else {
                Tensor::<B, 4usize, Int>::zeros(lhs.shape(), &self.device).bool()
            };
            output
        }
        ");
    }

    #[test]
    fn test_and_shape_forward() {
        let node = AndNodeBuilder::new("and1")
            .input_shape("lhs", 3)
            .input_shape("rhs", 3)
            .output_shape("output", 3)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: [i64; 3], rhs: [i64; 3]) -> [i64; 3] {
            let output = {
                let mut result = lhs;
                for (result_item, rhs_item) in result.iter_mut().zip(rhs.iter()) {
                    *result_item = if *result_item != 0 && *rhs_item != 0 { 1i64 } else { 0i64 };
                }
                result
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
        pub fn forward(
            &self,
            lhs: Tensor<B, 2, Bool>,
            rhs: Tensor<B, 2, Bool>,
        ) -> Tensor<B, 2, Bool> {
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
        pub fn forward(&self, lhs: [i64; 1], rhs: Tensor<B, 1, Bool>) -> Tensor<B, 1, Bool> {
            let output = Tensor::<
                B,
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
        pub fn forward(&self, lhs: Tensor<B, 1, Bool>, rhs: [i64; 1]) -> Tensor<B, 1, Bool> {
            let output = lhs
                .bool_and(
                    Tensor::<
                        B,
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
}
