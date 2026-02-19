use super::prelude::*;

impl NodeCodegen for onnx_ir::where_op::WhereNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let condition_arg = &self.inputs[0];
        let x_arg = &self.inputs[1];
        let y_arg = &self.inputs[2];
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        match &output_arg.ty {
            ArgType::Tensor(out_tensor) => {
                let broadcast_rank = out_tensor.rank;
                let target_dtype = out_tensor.dtype;

                // Get condition as tensor (condition uses Bool dtype, not target dtype)
                let cond = where_input_as_tensor(condition_arg, broadcast_rank, DType::Bool, scope);

                // Get y as tensor
                let y_tensor = where_input_as_tensor(y_arg, broadcast_rank, target_dtype, scope);

                // Check if x is a scalar - if so, use mask_fill
                if let ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) = &x_arg.ty {
                    // Convert x to a native scalar for mask_fill
                    let x_value = match &x_arg.ty {
                        ArgType::ScalarTensor(dtype) => {
                            let tensor = scope.arg(x_arg);
                            on_device_to_native(tensor, dtype)
                        }
                        _ => {
                            let name = arg_to_ident(x_arg);
                            quote! { #name }
                        }
                    };
                    // When y is also scalar, it becomes a [1,1,...,1] tensor that may be
                    // smaller than the condition. Expand y to match condition shape so
                    // mask_fill can broadcast the mask correctly.
                    if matches!(
                        &y_arg.ty,
                        ArgType::ScalarNative(_) | ArgType::ScalarTensor(_)
                    ) {
                        quote! {
                            let #output = {
                                let cond = #cond;
                                #y_tensor.expand(cond.dims()).mask_fill(cond, #x_value)
                            };
                        }
                    } else {
                        quote! {
                            let #output = #y_tensor.mask_fill(#cond, #x_value);
                        }
                    }
                } else {
                    // x is tensor or shape - use mask_where
                    let x_tensor =
                        where_input_as_tensor(x_arg, broadcast_rank, target_dtype, scope);
                    quote! {
                        let #output = #y_tensor.mask_where(#cond, #x_tensor);
                    }
                }
            }
            ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => {
                // Scalar output means all inputs are scalars
                let cond_expr = match &condition_arg.ty {
                    ArgType::ScalarTensor(dtype) => {
                        let tensor = scope.arg(condition_arg);
                        on_device_to_native(tensor, dtype)
                    }
                    _ => {
                        let name = arg_to_ident(condition_arg);
                        quote! { #name }
                    }
                };
                let x_expr = if x_arg.ty.is_on_device() {
                    scope.arg(x_arg)
                } else {
                    let name = arg_to_ident(x_arg);
                    quote! { #name }
                };
                let y_expr = if y_arg.ty.is_on_device() {
                    scope.arg(y_arg)
                } else {
                    let name = arg_to_ident(y_arg);
                    quote! { #name }
                };

                quote! {
                    let #output = if #cond_expr {
                        #x_expr
                    } else {
                        #y_expr
                    };
                }
            }
            ArgType::Shape(_) => {
                // Shape output - handle element-wise or whole shape selection
                match (&condition_arg.ty, &x_arg.ty, &y_arg.ty) {
                    (ArgType::Shape(_), ArgType::Shape(_), ArgType::Shape(_)) => {
                        // Element-wise selection between shape dimensions
                        let cond_name = arg_to_ident(condition_arg);
                        let x_name = arg_to_ident(x_arg);
                        let y_name = arg_to_ident(y_arg);

                        quote! {
                            let #output = {
                                let mut result = #y_name;
                                for (i, (cond_item, x_item)) in #cond_name.iter().zip(#x_name.iter()).enumerate() {
                                    if *cond_item != 0 {
                                        result[i] = *x_item;
                                    }
                                }
                                result
                            };
                        }
                    }
                    (
                        ArgType::ScalarNative(_) | ArgType::ScalarTensor(_),
                        ArgType::Shape(_),
                        ArgType::Shape(_),
                    ) => {
                        // Scalar condition: select entire shape x or y
                        let cond_expr = match &condition_arg.ty {
                            ArgType::ScalarTensor(dtype) => {
                                let tensor = scope.arg(condition_arg);
                                on_device_to_native(tensor, dtype)
                            }
                            _ => {
                                let name = arg_to_ident(condition_arg);
                                quote! { #name }
                            }
                        };
                        let x_name = arg_to_ident(x_arg);
                        let y_name = arg_to_ident(y_arg);

                        quote! {
                            let #output = if #cond_expr { #x_name } else { #y_name };
                        }
                    }
                    _ => panic!(
                        "Where with Shape output only supports: \
                         (Shape, Shape, Shape) for element-wise selection or \
                         (Scalar, Shape, Shape) for whole shape selection"
                    ),
                }
            }
        }
    }
}

// Helper function to convert an input to a tensor for broadcasting
fn where_input_as_tensor(
    arg: &Argument,
    broadcast_rank: usize,
    target_dtype: DType,
    scope: &mut super::super::scope::ScopeAtPosition<'_>,
) -> TokenStream {
    match &arg.ty {
        ArgType::Tensor(t) => {
            let tensor = scope.arg(arg);
            let rank = t.rank;

            if rank < broadcast_rank {
                // Unsqueeze leading dims to match broadcast rank
                let dims_to_unsqueeze: Vec<isize> =
                    (0..broadcast_rank - rank).map(|d| d as isize).collect();
                quote! { #tensor.unsqueeze_dims(&[#(#dims_to_unsqueeze),*]) }
            } else {
                tensor
            }
        }
        ArgType::ScalarTensor(_) => {
            // ScalarTensor is already a Tensor<B, 1> on device, just reshape/unsqueeze
            let tensor = scope.arg(arg);
            if broadcast_rank > 1 {
                let dims_to_unsqueeze: Vec<isize> =
                    (0..broadcast_rank - 1).map(|d| d as isize).collect();
                quote! { #tensor.unsqueeze_dims(&[#(#dims_to_unsqueeze),*]) }
            } else {
                tensor
            }
        }
        ArgType::ScalarNative(input_dtype) => {
            // Convert native scalar to tensor with shape [1, 1, ...] (broadcast_rank dimensions)
            let name = arg_to_ident(arg);
            let shape_vec: Vec<_> = (0..broadcast_rank).map(|_| quote! { 1 }).collect();
            let dtype_tokens = target_dtype.to_tokens();

            if target_dtype.is_float() {
                quote! {
                    Tensor::<B, 1>::from_data_dtype(
                        burn::tensor::TensorData::from([#name as f64]),
                        &*self.device,
                        #dtype_tokens
                    ).reshape([#(#shape_vec),*])
                }
            } else if target_dtype.is_int() || target_dtype.is_uint() {
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from([#name as i64]),
                        &*self.device,
                        #dtype_tokens
                    ).reshape([#(#shape_vec),*])
                }
            } else {
                // Bool output: if input is already bool, use directly; otherwise != 0
                let bool_expr = if input_dtype.is_bool() {
                    quote! { #name }
                } else {
                    quote! { #name != 0 }
                };
                quote! {
                    Tensor::<B, 1, burn::tensor::Bool>::from_data_dtype(
                        burn::tensor::TensorData::from([#bool_expr]),
                        &*self.device,
                        #dtype_tokens
                    ).reshape([#(#shape_vec),*])
                }
            }
        }
        ArgType::Shape(_) => {
            // Convert shape to tensor (rank 1) with explicit dtype to match target tensor
            let name = arg_to_ident(arg);
            let dtype_tokens = target_dtype.to_tokens();
            let tensor = quote! {
                Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                    burn::tensor::TensorData::from(&#name as &[i64]),
                    &*self.device,
                    #dtype_tokens
                )
            };

            if broadcast_rank > 1 {
                // Unsqueeze to match broadcast rank
                let dims_to_unsqueeze: Vec<isize> =
                    (0..broadcast_rank - 1).map(|d| d as isize).collect();
                quote! { #tensor.unsqueeze_dims(&[#(#dims_to_unsqueeze),*]) }
            } else {
                tensor
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::where_op::WhereNodeBuilder;

    #[test]
    fn test_where_tensor_tensor_tensor() {
        let node = WhereNodeBuilder::new("where1")
            .input_tensor("condition", 2, DType::Bool)
            .input_tensor("x", 2, DType::F32)
            .input_tensor("y", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            condition: Tensor<B, 2, Bool>,
            x: Tensor<B, 2>,
            y: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = y.mask_where(condition, x);
            output
        }
        ");
    }

    #[test]
    fn test_where_tensor_tensor_broadcasted_tensor_broadcasted() {
        let node = WhereNodeBuilder::new("where1")
            .input_tensor("condition", 2, DType::Bool)
            .input_tensor("x", 1, DType::F32)
            .input_tensor("y", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            condition: Tensor<B, 2, Bool>,
            x: Tensor<B, 1>,
            y: Tensor<B, 1>,
        ) -> Tensor<B, 2> {
            let output = y
                .unsqueeze_dims(&[0isize])
                .mask_where(condition, x.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_where_tensor_scalar_tensor() {
        let node = WhereNodeBuilder::new("where1")
            .input_tensor("condition", 2, DType::Bool)
            .input_scalar("x", DType::F32)
            .input_tensor("y", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            condition: Tensor<B, 2, Bool>,
            x: f32,
            y: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = y.mask_fill(condition, x);
            output
        }
        ");
    }

    #[test]
    fn test_where_tensor_scalar_scalar() {
        let node = WhereNodeBuilder::new("where1")
            .input_tensor("condition", 2, DType::Bool)
            .input_scalar("x", DType::F32)
            .input_scalar("y", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, condition: Tensor<B, 2, Bool>, x: f32, y: f32) -> Tensor<B, 2> {
            let output = {
                let cond = condition;
                Tensor::<
                    B,
                    1,
                >::from_data_dtype(
                        burn::tensor::TensorData::from([y as f64]),
                        &*self.device,
                        burn::tensor::DType::F32,
                    )
                    .reshape([1, 1])
                    .expand(cond.dims())
                    .mask_fill(cond, x)
            };
            output
        }
        ");
    }

    #[test]
    fn test_where_scalar_scalar_scalar() {
        let node = WhereNodeBuilder::new("where1")
            .input_scalar("condition", DType::Bool)
            .input_scalar("x", DType::F32)
            .input_scalar("y", DType::F32)
            .output_scalar("output", DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, condition: bool, x: f32, y: f32) -> f32 {
            let output = if condition { x } else { y };
            output
        }
        ");
    }
}
