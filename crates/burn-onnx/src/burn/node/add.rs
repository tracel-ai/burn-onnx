use super::prelude::*;

impl NodeCodegen for onnx_ir::node::arithmetic::AddNode {
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
                    quote! { #lhs.add(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.add(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).add(#rhs) }
                }
            }
            (lhs_ty, ArgType::ScalarNative(_)) if lhs_ty.is_on_device() => {
                quote! { #lhs.add_scalar(#rhs) }
            }
            (ArgType::ScalarNative(_), rhs_ty) if rhs_ty.is_on_device() => {
                quote! { #rhs.add_scalar(#lhs) }
            }
            (ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => quote! { #lhs + #rhs },
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = result_item.saturating_add(*rhs_item);
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
                            *result_item = result_item.saturating_add(__scalar as i64);
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
                            *result_item = result_item.saturating_add(__scalar as i64);
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
                    ).add(#rhs)
                }
            }
            (lhs_ty, ArgType::Shape(_)) if lhs_ty.is_on_device() => {
                let dtype_tokens = lhs_ty.elem_type().to_tokens();
                quote! {
                    #lhs.add(Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#rhs as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ))
                }
            }
            _ => unreachable!(
                "add: unsupported input types: {:?}, {:?}",
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
    use onnx_ir::node::arithmetic::{AddNode, AddNodeBuilder};

    fn create_add_node_tensor_tensor(name: &str, lhs_rank: usize, rhs_rank: usize) -> AddNode {
        AddNodeBuilder::new(name)
            .input_tensor("lhs", lhs_rank, DType::F32)
            .input_tensor("rhs", rhs_rank, DType::F32)
            .output_tensor("output", lhs_rank.max(rhs_rank), DType::F32)
            .build()
    }

    fn create_add_node_tensor_scalar(name: &str) -> AddNode {
        AddNodeBuilder::new(name)
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build()
    }

    #[test]
    fn test_add_forward_tensor_tensor() {
        let node = create_add_node_tensor_tensor("add1", 2, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.add(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_add_forward_tensor_scalar() {
        let node = create_add_node_tensor_scalar("add1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: f32) -> Tensor<B, 2> {
            let output = lhs.add_scalar(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_add_forward_broadcast() {
        let node = create_add_node_tensor_tensor("add1", 3, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 3>, rhs: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = lhs.add(rhs.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }
}
