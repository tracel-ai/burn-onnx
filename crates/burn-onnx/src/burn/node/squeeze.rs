use super::prelude::*;

impl NodeCodegen for onnx_ir::squeeze::SqueezeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        use onnx_ir::ir::ArgType;

        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        match (&input_arg.ty, &output_arg.ty) {
            (ArgType::Tensor(_), ArgType::Tensor(output_tensor)) => {
                let input = scope.arg(input_arg);

                match &self.config.axes {
                    Some(onnx_ir::squeeze::SqueezeInput::Static(axes_vec)) => {
                        // Use squeeze_dims with specific axes
                        let axes_arg = axes_vec.to_tokens();
                        let output_rank = output_tensor.rank.to_tokens();
                        quote! {
                            let #output = #input.squeeze_dims::<#output_rank>(&#axes_arg);
                        }
                    }
                    Some(onnx_ir::squeeze::SqueezeInput::Runtime(runtime_ref)) => {
                        // Runtime axes: read the axes tensor at forward time,
                        // normalize negative axes relative to the input rank,
                        // and pass the resulting Vec<isize> to squeeze_dims.
                        // The output rank is still compile-time-known from
                        // type inference.
                        let axes_arg = &self.inputs[runtime_ref.input_index];
                        let axes_expr = scope.arg(axes_arg);
                        let output_rank = output_tensor.rank.to_tokens();
                        let input_rank_lit = match &input_arg.ty {
                            ArgType::Tensor(t) => t.rank,
                            _ => unreachable!(),
                        };
                        let input_rank_tokens = input_rank_lit.to_tokens();
                        let raw_axes_bind = match &axes_arg.ty {
                            ArgType::Shape(_) => quote! {
                                let __raw_axes: alloc::vec::Vec<i64> =
                                    #axes_expr.iter().copied().collect();
                            },
                            _ => quote! {
                                let __raw_axes: alloc::vec::Vec<i64> = #axes_expr
                                    .to_data()
                                    .convert::<i64>()
                                    .into_vec::<i64>()
                                    .unwrap();
                            },
                        };
                        quote! {
                            let #output = {
                                #raw_axes_bind
                                let __rank: i64 = #input_rank_tokens;
                                let __axes: alloc::vec::Vec<isize> = __raw_axes
                                    .into_iter()
                                    .map(|v| (if v < 0 { v + __rank } else { v }) as isize)
                                    .collect();
                                #input.squeeze_dims::<#output_rank>(&__axes)
                            };
                        }
                    }
                    None => {
                        // When axes is None, squeeze all dimensions with size 1
                        let output_rank = output_tensor.rank.to_tokens();
                        quote! {
                            let #output = #input.squeeze::<#output_rank>();
                        }
                    }
                }
            }
            (ArgType::Shape(_), ArgType::ScalarNative(elem_type)) => {
                let input_name = arg_to_ident(input_arg);
                let cast_expr = shape_to_native(quote! { #input_name }, elem_type);
                quote! {
                    let #output = #cast_expr;
                }
            }
            (ArgType::Shape(_), ArgType::Shape(_)) => {
                // Shape(n) where n > 1 remains unchanged (squeeze is a no-op)
                let input_name = arg_to_ident(input_arg);

                quote! {
                    let #output = #input_name;
                }
            }
            (ArgType::ScalarNative(_), ArgType::ScalarNative(_))
            | (ArgType::ScalarTensor(_), ArgType::ScalarTensor(_)) => {
                // Scalar squeeze is a no-op
                let input_name = arg_to_ident(input_arg);

                quote! {
                    let #output = #input_name;
                }
            }
            (ArgType::ScalarTensor(_), ArgType::ScalarNative(elem_type)) => {
                let input = scope.arg(input_arg);
                let extract = on_device_to_native(input, elem_type);
                quote! {
                    let #output = #extract;
                }
            }
            (ArgType::Tensor(_), ArgType::ScalarTensor(_)) => {
                // Keep as Tensor<1> on device (no GPU stall)
                let input = scope.arg(input_arg);
                quote! {
                    let #output = #input.reshape([1]);
                }
            }
            (ArgType::Tensor(_), ArgType::ScalarNative(elem_type)) => {
                let input = scope.arg(input_arg);
                let extract = on_device_to_native(input, elem_type);
                quote! {
                    let #output = #extract;
                }
            }
            _ => panic!(
                "Squeeze: unsupported input/output combination: {:?} -> {:?}",
                input_arg.ty, output_arg.ty
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::squeeze::{SqueezeConfig, SqueezeInput, SqueezeNode, SqueezeNodeBuilder};

    fn create_squeeze_node_static(name: &str, axes: Vec<i64>) -> SqueezeNode {
        let config = SqueezeConfig {
            axes: Some(SqueezeInput::Static(axes)),
        };

        SqueezeNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_squeeze_forward_static_axes() {
        let node = create_squeeze_node_static("squeeze1", vec![1]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<2> {
            let output = input.squeeze_dims::<2>(&[1]);
            output
        }
        ");
    }

    #[test]
    fn test_squeeze_forward_multiple_axes() {
        let node = create_squeeze_node_static("squeeze1", vec![0, 2]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<2> {
            let output = input.squeeze_dims::<2>(&[0, 2]);
            output
        }
        ");
    }

    #[test]
    fn test_squeeze_forward_all_axes() {
        let config = SqueezeConfig { axes: None };
        let node = SqueezeNodeBuilder::new("squeeze1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<1> {
            let output = input.squeeze::<1>();
            output
        }
        ");
    }

    #[test]
    fn test_squeeze_runtime_axes_tensor() {
        use onnx_ir::ir::RuntimeInputRef;
        let config = SqueezeConfig {
            axes: Some(SqueezeInput::Runtime(RuntimeInputRef::new(
                "axes".to_string(),
                1,
            ))),
        };
        let node = SqueezeNodeBuilder::new("squeeze_rt")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("axes", 1, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, axes: Tensor<1, Int>) -> Tensor<2> {
            let output = {
                let __raw_axes: alloc::vec::Vec<i64> = axes
                    .to_data()
                    .convert::<i64>()
                    .into_vec::<i64>()
                    .unwrap();
                let __rank: i64 = 3;
                let __axes: alloc::vec::Vec<isize> = __raw_axes
                    .into_iter()
                    .map(|v| (if v < 0 { v + __rank } else { v }) as isize)
                    .collect();
                input.squeeze_dims::<2>(&__axes)
            };
            output
        }
        ");
    }

    #[test]
    fn test_squeeze_runtime_axes_shape() {
        use onnx_ir::ir::RuntimeInputRef;
        let config = SqueezeConfig {
            axes: Some(SqueezeInput::Runtime(RuntimeInputRef::new(
                "axes".to_string(),
                1,
            ))),
        };
        let node = SqueezeNodeBuilder::new("squeeze_rt_shape")
            .input_tensor("input", 3, DType::F32)
            .input_shape("axes", 1)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, axes: [i64; 1]) -> Tensor<2> {
            let output = {
                let __raw_axes: alloc::vec::Vec<i64> = axes.iter().copied().collect();
                let __rank: i64 = 3;
                let __axes: alloc::vec::Vec<isize> = __raw_axes
                    .into_iter()
                    .map(|v| (if v < 0 { v + __rank } else { v }) as isize)
                    .collect();
                input.squeeze_dims::<2>(&__axes)
            };
            output
        }
        ");
    }
}
