use super::prelude::*;

impl NodeCodegen for onnx_ir::imputer::ImputerNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = &self.inputs[0];
        let output = arg_to_ident(&self.outputs[0]);
        let input = scope.arg(input_arg);

        let function = match &input_arg.ty {
            ArgType::ScalarNative(scalar_ty) => {
                // Handle scalar inputs
                match scalar_ty {
                    DType::F32 | DType::F64 => {
                        if let Some(imputed_floats) = &self.config.imputed_value_floats {
                            if let Some(&first_value) = imputed_floats.first() {
                                let replaced_value =
                                    self.config.replaced_value_float.unwrap_or(f32::NAN);

                                // Generate appropriate literal type based on dtype
                                let first_literal = if scalar_ty == &DType::F64 {
                                    let first_f64 = first_value as f64;
                                    quote! { #first_f64 }
                                } else {
                                    quote! { #first_value }
                                };

                                if replaced_value.is_nan() {
                                    quote! { if #input.is_nan() { #first_literal } else { #input } }
                                } else {
                                    // Only generate replaced_literal for non-NaN values
                                    let replaced_literal = if scalar_ty == &DType::F64 {
                                        let replaced_f64 = replaced_value as f64;
                                        quote! { #replaced_f64 }
                                    } else {
                                        quote! { #replaced_value }
                                    };
                                    quote! { if #input == #replaced_literal { #first_literal } else { #input } }
                                }
                            } else {
                                quote! { #input }
                            }
                        } else {
                            quote! { #input }
                        }
                    }
                    _ => quote! { #input },
                }
            }
            ArgType::Tensor(tensor_ty) => {
                // Handle tensor inputs based on dtype
                match tensor_ty.dtype {
                    DType::F32 | DType::F64 => {
                        if let Some(imputed_floats) = &self.config.imputed_value_floats {
                            let replaced_value =
                                self.config.replaced_value_float.unwrap_or(f32::NAN);

                            if imputed_floats.len() == 1 {
                                let imputed_value = imputed_floats[0];
                                if replaced_value.is_nan() {
                                    quote! {
                                        {
                                            let mask = #input.clone().is_nan();
                                            #input.clone().mask_fill(mask, #imputed_value)
                                        }
                                    }
                                } else {
                                    quote! {
                                        {
                                            let mask = #input.clone().equal_elem(#replaced_value);
                                            #input.clone().mask_fill(mask, #imputed_value)
                                        }
                                    }
                                }
                            } else {
                                // Multiple imputed values per feature.
                                // Build a feature vector tensor and broadcast it over the input shape,
                                // then apply mask_where element-wise to avoid cross-feature mask bleed.
                                let imputed_values_vec: Vec<_> = imputed_floats.to_vec();
                                let imputed_values_len = imputed_values_vec.len();
                                let dtype_tokens = tensor_ty.dtype.to_tokens();
                                let reshape_dims: Vec<_> = (0..tensor_ty.rank.saturating_sub(1))
                                    .map(|_| quote! { 1usize })
                                    .chain(std::iter::once(quote! { #imputed_values_len }))
                                    .collect();
                                if replaced_value.is_nan() {
                                    quote! {
                                        {
                                            let mask = #input.clone().is_nan();
                                            let imputed_values = Tensor::<B, 1>::from_data(
                                                burn::tensor::TensorData::from([#((#imputed_values_vec) as f64),*]),
                                                (&self.device, #dtype_tokens),
                                            )
                                            .reshape([#(#reshape_dims),*])
                                            .expand(#input.dims());

                                            #input.clone().mask_where(mask, imputed_values)
                                        }
                                    }
                                } else {
                                    quote! {
                                        {
                                            let mask = #input.clone().equal_elem(#replaced_value);
                                            let imputed_values = Tensor::<B, 1>::from_data(
                                                burn::tensor::TensorData::from([#((#imputed_values_vec) as f64),*]),
                                                (&self.device, #dtype_tokens),
                                            )
                                            .reshape([#(#reshape_dims),*])
                                            .expand(#input.dims());

                                            #input.clone().mask_where(mask, imputed_values)
                                        }
                                    }
                                }
                            }
                        } else {
                            quote! { #input.clone() }
                        }
                    }
                    DType::I32 | DType::I64 | DType::I8 | DType::I16 => {
                        if let Some(imputed_ints) = &self.config.imputed_value_int64s {
                            let replaced_int = self.config.replaced_value_int64.unwrap_or(0i64);
                            if imputed_ints.len() == 1 {
                                let imputed_value = imputed_ints[0];
                                quote! {
                                    {
                                        let mask = #input.clone().equal_elem(#replaced_int);
                                        #input.clone().mask_fill(mask, #imputed_value)
                                    }
                                }
                            } else {
                                // Multiple imputed values per feature.
                                // Build a feature vector tensor and broadcast it over the input shape,
                                // then apply mask_where element-wise to avoid cross-feature mask bleed.
                                let imputed_values_vec: Vec<_> = imputed_ints.to_vec();
                                let imputed_values_len = imputed_values_vec.len();
                                let dtype_tokens = tensor_ty.dtype.to_tokens();
                                let reshape_dims: Vec<_> = (0..tensor_ty.rank.saturating_sub(1))
                                    .map(|_| quote! { 1usize })
                                    .chain(std::iter::once(quote! { #imputed_values_len }))
                                    .collect();
                                quote! {
                                    {
                                        let mask = #input.clone().equal_elem(#replaced_int);
                                        let imputed_values = Tensor::<B, 1, burn::tensor::Int>::from_data(
                                            burn::tensor::TensorData::from([#(#imputed_values_vec),*]),
                                            (&self.device, #dtype_tokens),
                                        )
                                        .reshape([#(#reshape_dims),*])
                                        .expand(#input.dims());

                                        #input.clone().mask_where(mask, imputed_values)
                                    }
                                }
                            }
                        } else {
                            quote! { #input.clone() }
                        }
                    }
                    _ => quote! { #input.clone() },
                }
            }
            _ => quote! { #input.clone() },
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
    use onnx_ir::imputer::ImputerConfig;
    use onnx_ir::imputer::ImputerNode;
    use onnx_ir::ir::{ArgType, TensorType};

    #[test]
    fn test_imputer_single_float() {
        let config = ImputerConfig::new(Some(vec![0.0]), None, None, None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ImputerNode::new("imputer1".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let mask = input.clone().is_nan();
                input.clone().mask_fill(mask, 0f32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_imputer_single_float_with_replaced_value() {
        let config = ImputerConfig::new(Some(vec![1.0]), None, Some(-999.0), None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ImputerNode::new("imputer2".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let mask = input.clone().equal_elem(-999f32);
                input.clone().mask_fill(mask, 1f32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_imputer_multiple_floats() {
        let config = ImputerConfig::new(Some(vec![0.0, 1.0, 2.0]), None, None, None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ImputerNode::new("imputer3".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let mask = input.clone().is_nan();
                let imputed_values = Tensor::<
                    B,
                    1,
                >::from_data(
                        burn::tensor::TensorData::from([
                            (0f32) as f64,
                            (1f32) as f64,
                            (2f32) as f64,
                        ]),
                        (&self.device, burn::tensor::DType::F32),
                    )
                    .reshape([1usize, 3usize])
                    .expand(input.dims());
                input.clone().mask_where(mask, imputed_values)
            };
            output
        }
        ");
    }

    #[test]
    fn test_imputer_single_int() {
        let config = ImputerConfig::new(None, Some(vec![0]), None, Some(-1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::I64, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::I64, 2, None)),
        );
        let node = ImputerNode::new("imputer4".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
            let output = {
                let mask = input.clone().equal_elem(-1i64);
                input.clone().mask_fill(mask, 0i64)
            };
            output
        }
        ");
    }

    #[test]
    fn test_imputer_int_default_replaced_value() {
        // replaced_value_int64 absent → default sentinel is 0 per ONNX spec
        let config = ImputerConfig::new(None, Some(vec![99]), None, None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::I64, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::I64, 2, None)),
        );
        let node = ImputerNode::new("imputer7".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
            let output = {
                let mask = input.clone().equal_elem(0i64);
                input.clone().mask_fill(mask, 99i64)
            };
            output
        }
        ");
    }

    #[test]
    fn test_imputer_scalar_float_nan_replacement() {
        // replaced_value_float absent → replace NaN
        let config = ImputerConfig::new(Some(vec![0.0f32]), None, None, None);
        let input = onnx_ir::ir::Argument::new("input", ArgType::ScalarNative(DType::F32));
        let output = onnx_ir::ir::Argument::new("output", ArgType::ScalarNative(DType::F32));
        let node = ImputerNode::new("imputer5".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: f32) -> f32 {
            let output = if input.is_nan() { 0f32 } else { input };
            output
        }
        ");
    }

    #[test]
    fn test_imputer_scalar_float_sentinel_replacement() {
        // replaced_value_float set to a non-NaN sentinel → only replace that value
        let config = ImputerConfig::new(Some(vec![1.0f32]), None, Some(-999.0f32), None);
        let input = onnx_ir::ir::Argument::new("input", ArgType::ScalarNative(DType::F32));
        let output = onnx_ir::ir::Argument::new("output", ArgType::ScalarNative(DType::F32));
        let node = ImputerNode::new("imputer6".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: f32) -> f32 {
            let output = if input == -999f32 { 1f32 } else { input };
            output
        }
        ");
    }
}
