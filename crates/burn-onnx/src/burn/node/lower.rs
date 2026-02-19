use super::prelude::*;

impl NodeCodegen for onnx_ir::comparison::LessNode {
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

                if lhs_rank == rhs_rank {
                    quote! { #lhs_value.lower(#rhs_value) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.lower(#rhs_value.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.unsqueeze_dims(&[#(#dims),*]).lower(#rhs_value) }
                }
            }
            (lhs_ty, ArgType::ScalarNative(_)) if lhs_ty.is_on_device() => {
                quote! { #lhs_value.lower_elem(#rhs_value) }
            }
            (ArgType::ScalarNative(_), rhs_ty) if rhs_ty.is_on_device() => {
                // L < R == R > L
                quote! { #rhs_value.greater_elem(#lhs_value) }
            }
            (ArgType::Shape(_), rhs_ty) if rhs_ty.is_on_device() => {
                let dtype_tokens = rhs_ty.elem_type().to_tokens();
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#lhs_value as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ).lower(#rhs_value)
                }
            }
            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_on_device() => {
                let dtype_tokens = lhs_ty.elem_type().to_tokens();
                quote! {
                    #lhs_value.lower(Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#rhs_value as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ))
                }
            }
            (ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => {
                quote! { #lhs_value < #rhs_value }
            }
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs_value;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs_value.iter()) {
                        *result_item = if result_item < rhs_item { 1i64 } else { 0i64 };
                    }
                    result
                }
            },
            (lhs, rhs) => panic!("lower is not supported for {lhs:?} < {rhs:?}"),
        };

        quote! {
            let #output = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::comparison::LessNodeBuilder;

    // --- on_device + on_device ---

    #[test]
    fn test_tensor_tensor_same_rank() {
        let node = LessNodeBuilder::new("less1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = lhs.lower(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_tensor_tensor_broadcast_lhs_higher() {
        let node = LessNodeBuilder::new("less1")
            .input_tensor("lhs", 3, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 3, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 3>, rhs: Tensor<B, 2>) -> Tensor<B, 3, Bool> {
            let output = lhs.lower(rhs.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_tensor_tensor_broadcast_rhs_higher() {
        let node = LessNodeBuilder::new("less1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 3, DType::F32)
            .output_tensor("output", 3, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 3>) -> Tensor<B, 3, Bool> {
            let output = lhs.unsqueeze_dims(&[0isize]).lower(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_tensor_scalar_tensor() {
        let node = LessNodeBuilder::new("less1")
            .input_tensor("lhs", 3, DType::F32)
            .input_scalar_tensor("rhs", DType::F32)
            .output_tensor("output", 3, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 3>, rhs: Tensor<B, 1>) -> Tensor<B, 3, Bool> {
            let output = lhs.lower(rhs.unsqueeze_dims(&[0isize, 1isize]));
            output
        }
        ");
    }

    #[test]
    fn test_scalar_tensor_tensor() {
        let node = LessNodeBuilder::new("less1")
            .input_scalar_tensor("lhs", DType::F32)
            .input_tensor("rhs", 3, DType::F32)
            .output_tensor("output", 3, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 1>, rhs: Tensor<B, 3>) -> Tensor<B, 3, Bool> {
            let output = lhs.unsqueeze_dims(&[0isize, 1isize]).lower(rhs);
            output
        }
        ");
    }

    // --- on_device + ScalarNative ---

    #[test]
    fn test_tensor_scalar_native() {
        let node = LessNodeBuilder::new("less1")
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: f32) -> Tensor<B, 2, Bool> {
            let output = lhs.lower_elem(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_scalar_native_tensor() {
        let node = LessNodeBuilder::new("less1")
            .input_scalar("lhs", DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: f32, rhs: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = rhs.greater_elem(lhs);
            output
        }
        ");
    }

    // --- Shape + on_device ---

    #[test]
    fn test_shape_tensor() {
        let node = LessNodeBuilder::new("less1")
            .input_shape("lhs", 4)
            .input_tensor("rhs", 1, DType::I64)
            .output_tensor("output", 1, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: Tensor<B, 1, Int>) -> Tensor<B, 1, Bool> {
            let output = Tensor::<
                B,
                1,
                burn::tensor::Int,
            >::from_data_dtype(
                    burn::tensor::TensorData::from(&lhs as &[i64]),
                    &*self.device,
                    burn::tensor::DType::I64,
                )
                .lower(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_tensor_shape() {
        let node = LessNodeBuilder::new("less1")
            .input_tensor("lhs", 1, DType::I64)
            .input_shape("rhs", 4)
            .output_tensor("output", 1, DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<B, 1, Int>, rhs: [i64; 4]) -> Tensor<B, 1, Bool> {
            let output = lhs
                .lower(
                    Tensor::<
                        B,
                        1,
                        burn::tensor::Int,
                    >::from_data_dtype(
                        burn::tensor::TensorData::from(&rhs as &[i64]),
                        &*self.device,
                        burn::tensor::DType::I64,
                    ),
                );
            output
        }
        ");
    }

    // --- ScalarNative + ScalarNative ---

    #[test]
    fn test_scalar_native_scalar_native() {
        let node = LessNodeBuilder::new("less1")
            .input_scalar("lhs", DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_scalar("output", DType::Bool)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: f32, rhs: f32) -> bool {
            let output = lhs < rhs;
            output
        }
        ");
    }

    // --- Shape + Shape ---

    #[test]
    fn test_shape_shape() {
        let node = LessNodeBuilder::new("less1")
            .input_shape("lhs", 4)
            .input_shape("rhs", 4)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: [i64; 4]) -> [i64; 4] {
            let output = {
                let mut result = lhs;
                for (result_item, rhs_item) in result.iter_mut().zip(rhs.iter()) {
                    *result_item = if result_item < rhs_item { 1i64 } else { 0i64 };
                }
                result
            };
            output
        }
        ");
    }
}
