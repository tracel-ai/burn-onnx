use super::prelude::*;

impl NodeCodegen for onnx_ir::node::arithmetic::DivNode {
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

                if lhs_rank == rhs_rank {
                    quote! { #lhs.div(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.div(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).div(#rhs) }
                }
            }
            (lhs_ty, ArgType::ScalarNative(_)) if lhs_ty.is_on_device() => {
                quote! { #lhs.div_scalar(#rhs) }
            }
            (ArgType::ScalarNative(dtype), rhs_ty) if rhs_ty.is_on_device() => {
                // Use the built-in Div impl: f32 / Tensor -> tensor.recip().mul_scalar(f32)
                if dtype.is_float() {
                    quote! { #lhs / #rhs }
                } else if dtype.is_int() || dtype.is_uint() {
                    let cast_type = match rhs_ty.elem_type() {
                        DType::F64 => quote! { f64 },
                        _ => quote! { f32 },
                    };
                    quote! { (#lhs as #cast_type) / #rhs }
                } else {
                    panic!("Unsupported scalar type for division: {:?}", dtype)
                }
            }
            (ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => quote! { #lhs / #rhs },
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = if *rhs_item != 0 { *result_item / *rhs_item } else { *result_item };
                    }
                    result
                }
            },
            (ArgType::Shape(_), rhs_ty) if rhs_ty.is_scalar() => {
                let scalar_expr = if rhs_ty.is_scalar_tensor() {
                    on_device_to_native(rhs.clone(), &rhs_ty.elem_type())
                } else {
                    quote! { #rhs as i64 }
                };
                quote! {
                    {
                        let mut result = #lhs;
                        let __scalar = #scalar_expr;
                        for result_item in result.iter_mut() {
                            *result_item = if __scalar as i64 != 0 { *result_item / (__scalar as i64) } else { *result_item };
                        }
                        result
                    }
                }
            }
            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_scalar() => {
                let scalar_expr = if lhs_ty.is_scalar_tensor() {
                    on_device_to_native(lhs.clone(), &lhs_ty.elem_type())
                } else {
                    quote! { #lhs as i64 }
                };
                quote! {
                    {
                        let mut result = #rhs;
                        let __scalar = #scalar_expr;
                        for result_item in result.iter_mut() {
                            *result_item = if *result_item != 0 { (__scalar as i64) / *result_item } else { (__scalar as i64) };
                        }
                        result
                    }
                }
            }
            (ArgType::Shape(_), rhs_ty) if rhs_ty.is_on_device() => {
                let dtype_tokens = rhs_ty.elem_type().to_tokens();
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#lhs as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ).div(#rhs)
                }
            }
            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_on_device() => {
                let dtype_tokens = lhs_ty.elem_type().to_tokens();
                quote! {
                    #lhs.div(Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#rhs as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ))
                }
            }
            _ => unreachable!(
                "div: unsupported input types: {:?}, {:?}",
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
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::arithmetic::DivNodeBuilder;

    #[test]
    fn test_div_forward_tensor_tensor() {
        let node = DivNodeBuilder::new("div1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.div(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_div_forward_tensor_scalar() {
        let node = DivNodeBuilder::new("div1")
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: f32) -> Tensor<B, 2> {
            let output = lhs.div_scalar(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_div_forward_scalar_tensor() {
        let node = DivNodeBuilder::new("div1")
            .input_scalar("lhs", DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: f32, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs / rhs;
            output
        }
        ");
    }
}
