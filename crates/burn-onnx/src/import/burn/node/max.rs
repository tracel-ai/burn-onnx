use super::prelude::*;

impl NodeCodegen for onnx_ir::node::max::MaxNode {
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
                quote! { #lhs_bc.max_pair(#rhs_bc) }
            }

            (ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => {
                quote! { #lhs.max(#rhs) }
            }
            (lhs_ty, ArgType::ScalarNative(_)) if lhs_ty.is_on_device() => {
                quote! { #lhs.clamp_min(#rhs) }
            }
            (ArgType::ScalarNative(_), rhs_ty) if rhs_ty.is_on_device() => {
                quote! { #rhs.clamp_min(#lhs) }
            }

            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = (*result_item).max(*rhs_item);
                    }
                    result
                }
            },

            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_scalar() => {
                let scalar_expr = scalar_as_i64(lhs_arg, lhs.clone());
                quote! {
                    {
                        let mut result = #rhs;
                        let __scalar = #scalar_expr;
                        for result_item in result.iter_mut() {
                            *result_item = (*result_item).max(__scalar);
                        }
                        result
                    }
                }
            }
            (ArgType::Shape(_), rhs_ty) if rhs_ty.is_scalar() => {
                let scalar_expr = scalar_as_i64(rhs_arg, rhs.clone());
                quote! {
                    {
                        let mut result = #lhs;
                        let __scalar = #scalar_expr;
                        for result_item in result.iter_mut() {
                            *result_item = (*result_item).max(__scalar);
                        }
                        result
                    }
                }
            }

            (lhs_ty, ArgType::Shape(_))
                if lhs_ty.is_on_device()
                    && (lhs_ty.elem_type().is_int() || lhs_ty.elem_type().is_uint()) =>
            {
                let dtype = lhs_ty.elem_type();
                let cast = if dtype == DType::I64 {
                    quote! {}
                } else {
                    let dtype_tokens = lhs_ty.elem_type().to_tokens();
                    quote! { .cast(#dtype_tokens) }
                };

                let rhs_tensor = quote! {
                    Tensor::<1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::from(&#rhs as &[i64]),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    #cast
                };
                let rhs_bc = broadcast_helpers::leading_broadcast(rhs_tensor, 1, lhs_ty.rank());
                quote! { #rhs_bc.max_pair(#lhs) }
            }
            (ArgType::Shape(_), rhs_ty)
                if rhs_ty.is_on_device()
                    && (rhs_ty.elem_type().is_int() || rhs_ty.elem_type().is_uint()) =>
            {
                let dtype = rhs_ty.elem_type();
                let cast = if dtype == DType::I64 {
                    quote! {}
                } else {
                    let dtype_tokens = rhs_ty.elem_type().to_tokens();
                    quote! { .cast(#dtype_tokens) }
                };

                let lhs_tensor = quote! {
                    Tensor::<1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::from(&#lhs as &[i64]),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    #cast
                };
                let lhs_bc = broadcast_helpers::leading_broadcast(lhs_tensor, 1, rhs_ty.rank());
                quote! { #lhs_bc.max_pair(#rhs) }
            }
            _ => unreachable!(
                "max: unsupported input types: lhs {:?}, rhs {:?}",
                lhs_arg.ty, rhs_arg.ty,
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
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::max::{MaxNode, MaxNodeBuilder};

    fn create_max_node(name: &str, lhs_rank: usize, rhs_rank: usize) -> MaxNode {
        MaxNodeBuilder::new(name)
            .input_tensor("a", lhs_rank, DType::F32)
            .input_tensor("b", rhs_rank, DType::F32)
            .output_tensor("output", lhs_rank.max(rhs_rank), DType::F32)
            .build()
    }

    // --- on_device + on_device ---

    #[test]
    fn test_tensor_tensor_same_rank() {
        let node = create_max_node("max1", 2, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
            let output = a.max_pair(b);
            output
        }
        ");
    }

    #[test]
    fn test_tensor_tensor_broadcast_lhs_higher() {
        let node = create_max_node("max1", 3, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<3>, b: Tensor<2>) -> Tensor<3> {
            let output = a.max_pair((b).unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_tensor_tensor_broadcast_rhs_higher() {
        let node = create_max_node("max1", 2, 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<2>, b: Tensor<3>) -> Tensor<3> {
            let output = (a).unsqueeze_dims(&[0isize]).max_pair(b);
            output
        }
        ");
    }

    #[test]
    fn test_tensor_scalar_tensor() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("lhs", 3, DType::F32)
            .input_scalar_tensor("rhs", DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<3>, rhs: Tensor<1>) -> Tensor<3> {
            let output = lhs.max_pair((rhs).unsqueeze_dims(&[0isize, 1isize]));
            output
        }
        ");
    }

    #[test]
    fn test_scalar_tensor_tensor() {
        let node = MaxNodeBuilder::new("max1")
            .input_scalar_tensor("lhs", DType::F32)
            .input_tensor("rhs", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<1>, rhs: Tensor<3>) -> Tensor<3> {
            let output = (lhs).unsqueeze_dims(&[0isize, 1isize]).max_pair(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_scalar_tensor_scalar_tensor() {
        let node = MaxNodeBuilder::new("max1")
            .input_scalar_tensor("lhs", DType::F32)
            .input_scalar_tensor("rhs", DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<1>, rhs: Tensor<1>) -> Tensor<1> {
            let output = lhs.max_pair(rhs);
            output
        }
        ");
    }

    // --- on_device + ScalarNative ---

    #[test]
    fn test_tensor_scalar_native() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<2>, rhs: f32) -> Tensor<2> {
            let output = lhs.clamp_min(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_scalar_native_tensor() {
        let node = MaxNodeBuilder::new("max1")
            .input_scalar("lhs", DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: f32, rhs: Tensor<2>) -> Tensor<2> {
            let output = rhs.clamp_min(lhs);
            output
        }
        ");
    }

    // --- ScalarNative + ScalarNative ---

    #[test]
    fn test_scalar_native_scalar_native() {
        let node = MaxNodeBuilder::new("max1")
            .input_scalar("lhs", DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_scalar("output", DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: f32, rhs: f32) -> f32 {
            let output = lhs.max(rhs);
            output
        }
        ");
    }

    // --- Shape + Shape ---

    #[test]
    fn test_shape_shape() {
        let node = MaxNodeBuilder::new("max1")
            .input_shape("lhs", 4)
            .input_shape("rhs", 4)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: [i64; 4]) -> [i64; 4] {
            let output = {
                let mut result = lhs;
                for (result_item, rhs_item) in result.iter_mut().zip(rhs.iter()) {
                    *result_item = (*result_item).max(*rhs_item);
                }
                result
            };
            output
        }
        ");
    }

    // --- Shape + Scalar ---

    #[test]
    fn test_shape_scalar_native() {
        let node = MaxNodeBuilder::new("max1")
            .input_shape("lhs", 4)
            .input_scalar("rhs", DType::I64)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: i64) -> [i64; 4] {
            let output = {
                let mut result = lhs;
                let __scalar = rhs as i64;
                for result_item in result.iter_mut() {
                    *result_item = (*result_item).max(__scalar);
                }
                result
            };
            output
        }
        ");
    }

    #[test]
    fn test_shape_scalar_tensor() {
        let node = MaxNodeBuilder::new("max1")
            .input_shape("lhs", 4)
            .input_scalar_tensor("rhs", DType::I64)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: Tensor<1, Int>) -> [i64; 4] {
            let output = {
                let mut result = lhs;
                let __scalar = (rhs).into_scalar::<i64>() as i64;
                for result_item in result.iter_mut() {
                    *result_item = (*result_item).max(__scalar);
                }
                result
            };
            output
        }
        ");
    }

    #[test]
    fn test_scalar_native_shape() {
        let node = MaxNodeBuilder::new("max1")
            .input_scalar("lhs", DType::I64)
            .input_shape("rhs", 4)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: i64, rhs: [i64; 4]) -> [i64; 4] {
            let output = {
                let mut result = rhs;
                let __scalar = lhs as i64;
                for result_item in result.iter_mut() {
                    *result_item = (*result_item).max(__scalar);
                }
                result
            };
            output
        }
        ");
    }

    #[test]
    fn test_scalar_tensor_shape() {
        let node = MaxNodeBuilder::new("max1")
            .input_scalar_tensor("lhs", DType::I64)
            .input_shape("rhs", 4)
            .output_shape("output", 4)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<1, Int>, rhs: [i64; 4]) -> [i64; 4] {
            let output = {
                let mut result = rhs;
                let __scalar = (lhs).into_scalar::<i64>() as i64;
                for result_item in result.iter_mut() {
                    *result_item = (*result_item).max(__scalar);
                }
                result
            };
            output
        }
        ");
    }

    // --- Shape + on_device ---

    #[test]
    fn test_shape_tensor() {
        let node = MaxNodeBuilder::new("max1")
            .input_shape("lhs", 4)
            .input_tensor("rhs", 3, DType::I32)
            .output_tensor("output", 3, DType::I32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: Tensor<3, Int>) -> Tensor<3, Int> {
            let output = (Tensor::<
                1,
                burn::tensor::Int,
            >::from_data(
                    burn::tensor::TensorData::from(&lhs as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                )
                .cast(burn::tensor::DType::I32))
                .unsqueeze_dims(&[0isize, 1isize])
                .max_pair(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_shape_tensor_no_cast() {
        let node = MaxNodeBuilder::new("max1")
            .input_shape("lhs", 4)
            .input_tensor("rhs", 3, DType::I64)
            .output_tensor("output", 3, DType::I64)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: [i64; 4], rhs: Tensor<3, Int>) -> Tensor<3, Int> {
            let output = (Tensor::<
                1,
                burn::tensor::Int,
            >::from_data(
                burn::tensor::TensorData::from(&lhs as &[i64]),
                (&self.device, burn::tensor::DType::I64),
            ))
                .unsqueeze_dims(&[0isize, 1isize])
                .max_pair(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_tensor_shape() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("lhs", 3, DType::I32)
            .input_shape("rhs", 4)
            .output_tensor("output", 3, DType::I32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<3, Int>, rhs: [i64; 4]) -> Tensor<3, Int> {
            let output = (Tensor::<
                1,
                burn::tensor::Int,
            >::from_data(
                    burn::tensor::TensorData::from(&rhs as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                )
                .cast(burn::tensor::DType::I32))
                .unsqueeze_dims(&[0isize, 1isize])
                .max_pair(lhs);
            output
        }
        ");
    }

    #[test]
    fn test_tensor_shape_no_cast() {
        let node = MaxNodeBuilder::new("max1")
            .input_tensor("lhs", 3, DType::I64)
            .input_shape("rhs", 4)
            .output_tensor("output", 3, DType::I64)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, lhs: Tensor<3, Int>, rhs: [i64; 4]) -> Tensor<3, Int> {
            let output = (Tensor::<
                1,
                burn::tensor::Int,
            >::from_data(
                burn::tensor::TensorData::from(&rhs as &[i64]),
                (&self.device, burn::tensor::DType::I64),
            ))
                .unsqueeze_dims(&[0isize, 1isize])
                .max_pair(lhs);
            output
        }
        ");
    }
}
