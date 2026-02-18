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
                    Some(onnx_ir::squeeze::SqueezeInput::Runtime(_)) => {
                        panic!("Runtime squeeze axes not yet supported in burn-onnx")
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
                // Keep as Tensor<B, 1> on device (no GPU stall)
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
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
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
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
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
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 1> {
            let output = input.squeeze::<1>();
            output
        }
        ");
    }
}
