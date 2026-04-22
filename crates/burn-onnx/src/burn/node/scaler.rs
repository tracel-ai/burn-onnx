use super::prelude::*;

impl NodeCodegen for onnx_ir::scaler::ScalerNode {
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

        // Formula: Y = (X - offset) * scale.
        // Input is always a tensor per ONNX spec (validated in onnx-ir processor).
        let function = match &input_arg.ty {
            ArgType::Tensor(tensor_type) => {
                // Output is always tensor(float) per ONNX spec; cast integer/double inputs to F32.

                let input_rank = tensor_type.rank;

                // Cast to F32 if input is not already F32 (integers or double)
                let input_expr = match tensor_type.dtype {
                    DType::F32 => quote! { #input.clone() },
                    DType::F64 => quote! { #input.cast(burn::tensor::DType::F32) },
                    _ => quote! { #input.float().cast(burn::tensor::DType::F32) },
                };

                // Helper to create reshape dimensions: [1, 1, ..., 1, num_features]
                let create_reshape_dims = |num_features: usize| -> Vec<TokenStream> {
                    (0..input_rank.saturating_sub(1))
                        .map(|_| quote! { 1usize })
                        .chain(std::iter::once(quote! { #num_features }))
                        .collect()
                };

                match (&self.config.offset, &self.config.scale) {
                    (Some(offset_values), Some(scale_values)) => {
                        let offset_values: Vec<_> = offset_values.to_vec();
                        let scale_values: Vec<_> = scale_values.to_vec();
                        let num_features = offset_values.len();
                        let reshape_dims = create_reshape_dims(num_features);

                        quote! {
                            {
                                let offset_tensor = Tensor::<B, 1>::from_data(
                                    [#(#offset_values),*],
                                    (&self.device, burn::tensor::DType::F32),
                                )
                                .reshape([#(#reshape_dims),*]);
                                let scale_tensor = Tensor::<B, 1>::from_data(
                                    [#(#scale_values),*],
                                    (&self.device, burn::tensor::DType::F32),
                                )
                                .reshape([#(#reshape_dims),*]);
                                (#input_expr - offset_tensor) * scale_tensor
                            }
                        }
                    }
                    (Some(offset_values), None) => {
                        let offset_values: Vec<_> = offset_values.to_vec();
                        let num_features = offset_values.len();
                        let reshape_dims = create_reshape_dims(num_features);

                        quote! {
                            {
                                let offset_tensor = Tensor::<B, 1>::from_data(
                                    [#(#offset_values),*],
                                    (&self.device, burn::tensor::DType::F32),
                                )
                                .reshape([#(#reshape_dims),*]);
                                #input_expr - offset_tensor
                            }
                        }
                    }
                    (None, Some(scale_values)) => {
                        let scale_values: Vec<_> = scale_values.to_vec();
                        let num_features = scale_values.len();
                        let reshape_dims = create_reshape_dims(num_features);

                        quote! {
                            {
                                let scale_tensor = Tensor::<B, 1>::from_data(
                                    [#(#scale_values),*],
                                    (&self.device, burn::tensor::DType::F32),
                                )
                                .reshape([#(#reshape_dims),*]);
                                #input_expr * scale_tensor
                            }
                        }
                    }
                    (None, None) => {
                        quote! { #input_expr }
                    }
                }
            }
            ty => {
                unreachable!("Scaler input is always a tensor (validated in onnx-ir), got {ty:?}")
            }
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
    use onnx_ir::ir::{ArgType, TensorType};
    use onnx_ir::scaler::ScalerConfig;
    use onnx_ir::scaler::ScalerNode;

    #[test]
    fn test_scaler_scale_only() {
        let config = ScalerConfig::new(Some(vec![2.0]), None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler1".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let scale_tensor = Tensor::<
                    B,
                    1,
                >::from_data([2f32], (&self.device, burn::tensor::DType::F32))
                    .reshape([1usize, 1usize]);
                input.clone() * scale_tensor
            };
            output
        }
        ");
    }

    #[test]
    fn test_scaler_offset_only() {
        let config = ScalerConfig::new(None, Some(vec![1.0]));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler2".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let offset_tensor = Tensor::<
                    B,
                    1,
                >::from_data([1f32], (&self.device, burn::tensor::DType::F32))
                    .reshape([1usize, 1usize]);
                input.clone() - offset_tensor
            };
            output
        }
        ");
    }

    #[test]
    fn test_scaler_both_scale_and_offset() {
        let config = ScalerConfig::new(Some(vec![2.0]), Some(vec![1.0]));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler3".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let offset_tensor = Tensor::<
                    B,
                    1,
                >::from_data([1f32], (&self.device, burn::tensor::DType::F32))
                    .reshape([1usize, 1usize]);
                let scale_tensor = Tensor::<
                    B,
                    1,
                >::from_data([2f32], (&self.device, burn::tensor::DType::F32))
                    .reshape([1usize, 1usize]);
                (input.clone() - offset_tensor) * scale_tensor
            };
            output
        }
        ");
    }

    #[test]
    fn test_scaler_no_transform() {
        let config = ScalerConfig::new(None, None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler4".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clone();
            output
        }
        ");
    }

    #[test]
    fn test_scaler_per_feature_scaling() {
        let config = ScalerConfig::new(Some(vec![1.0, 2.0, 3.0]), Some(vec![0.5, 1.0, 1.5]));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler5".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let offset_tensor = Tensor::<
                    B,
                    1,
                >::from_data([0.5f32, 1f32, 1.5f32], (&self.device, burn::tensor::DType::F32))
                    .reshape([1usize, 3usize]);
                let scale_tensor = Tensor::<
                    B,
                    1,
                >::from_data([1f32, 2f32, 3f32], (&self.device, burn::tensor::DType::F32))
                    .reshape([1usize, 3usize]);
                (input.clone() - offset_tensor) * scale_tensor
            };
            output
        }
        ");
    }

    #[test]
    fn test_scaler_int_input_both_scale_and_offset() {
        // Integer inputs must be cast to F32; output is always tensor(float) per ONNX spec
        let config = ScalerConfig::new(Some(vec![2.0]), Some(vec![1.0]));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::I64, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler6".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = {
                let offset_tensor = Tensor::<
                    B,
                    1,
                >::from_data([1f32], (&self.device, burn::tensor::DType::F32))
                    .reshape([1usize, 1usize]);
                let scale_tensor = Tensor::<
                    B,
                    1,
                >::from_data([2f32], (&self.device, burn::tensor::DType::F32))
                    .reshape([1usize, 1usize]);
                (input.float().cast(burn::tensor::DType::F32) - offset_tensor) * scale_tensor
            };
            output
        }
        ");
    }

    #[test]
    fn test_scaler_int_input_no_transform() {
        // Even with no scale/offset, integer input must be cast to F32
        let config = ScalerConfig::new(None, None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::I32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler7".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = input.float().cast(burn::tensor::DType::F32);
            output
        }
        ");
    }
}
