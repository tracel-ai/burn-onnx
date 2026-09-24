use super::prelude::*;

fn int_categories(config: &onnx_ir::one_hot_encoder::OneHotEncoderConfig) -> Vec<i64> {
    config.cats_int64s.clone().unwrap_or_default()
}

impl NodeCodegen for onnx_ir::one_hot_encoder::OneHotEncoderNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Store the category lookup table once at construction time instead of
        // rebuilding it on every forward() call — category lists can be large
        // (thousands of entries), and forward() may be called per-inference.
        let cats = int_categories(&self.config);
        let name = Ident::new(&self.name, Span::call_site());

        Some(Field::new(
            &self.name,
            quote! { Tensor<1, Int> },
            quote! {
                let #name: Tensor<1, Int> = Tensor::<1, Int>::from_data(
                    [#(#cats),*],
                    (device, burn::tensor::DType::I64),
                );
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = &self.inputs[0];
        let output = arg_to_ident(&self.outputs[0]);
        let input = scope.arg(input_arg);
        let field_name = Ident::new(&self.name, Span::call_site());

        let function = match &input_arg.ty {
            ArgType::Tensor(tensor_type) => {
                let input_rank = tensor_type.rank;

                let num_categories = int_categories(&self.config).len();

                // Build reshape dims for the category tensor to broadcast:
                // input shape: [d0, d1, ..., d_{r-1}]
                // category tensor shape: [1, 1, ..., 1, num_categories]
                // The category tensor has rank = input_rank + 1, with 1s for all input dims
                // and num_categories for the last dim.
                let ones: Vec<TokenStream> = (0..input_rank).map(|_| quote! { 1usize }).collect();

                // The spec casts float inputs to integers and looks them up in
                // cats_int64s, so every input compares in I64.
                let x = match tensor_type.dtype {
                    DType::I32 | DType::I64 => quote! { #input.cast(burn::tensor::DType::I64) },
                    DType::F32 | DType::F64 => {
                        quote! { #input.int().cast(burn::tensor::DType::I64) }
                    }
                    _ => unreachable!(
                        "OneHotEncoder input dtype is validated in onnx-ir; got {:?}",
                        tensor_type.dtype
                    ),
                };
                quote! {
                    {
                        let x = #x;
                        let x_unsqueezed = x.unsqueeze_dim(#input_rank);
                        let cats = self.#field_name.clone().reshape([#(#ones,)* #num_categories]);
                        x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
                    }
                }
            }
            ty => {
                unreachable!(
                    "OneHotEncoder input is always a tensor (validated in onnx-ir), got {ty:?}"
                )
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
    use onnx_ir::one_hot_encoder::OneHotEncoderConfig;
    use onnx_ir::one_hot_encoder::OneHotEncoderNode;

    #[test]
    fn test_onehotencoder_1d_input() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2, 3]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe1".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1>) -> Tensor<2> {
            let output = {
                let x = input.int().cast(burn::tensor::DType::I64);
                let x_unsqueezed = x.unsqueeze_dim(1usize);
                let cats = self.ohe1.clone().reshape([1usize, 4usize]);
                x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_onehotencoder_2d_input() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
        );
        let node = OneHotEncoderNode::new("ohe2".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<3> {
            let output = {
                let x = input.int().cast(burn::tensor::DType::I64);
                let x_unsqueezed = x.unsqueeze_dim(2usize);
                let cats = self.ohe2.clone().reshape([1usize, 1usize, 3usize]);
                x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_onehotencoder_int_input() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2, 3, 4]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::I64, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe3".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<1, Int>) -> Tensor<2> {
            let output = {
                let x = input.cast(burn::tensor::DType::I64);
                let x_unsqueezed = x.unsqueeze_dim(1usize);
                let cats = self.ohe3.clone().reshape([1usize, 5usize]);
                x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_onehotencoder_f64_input() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F64, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe4".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1>) -> Tensor<2> {
            let output = {
                let x = input.int().cast(burn::tensor::DType::I64);
                let x_unsqueezed = x.unsqueeze_dim(1usize);
                let cats = self.ohe4.clone().reshape([1usize, 3usize]);
                x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_onehotencoder_field_init() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2, 3]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe1".to_string(), vec![input], vec![output], config);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let ohe1: Tensor<1, Int> = Tensor::<
            1,
            Int,
        >::from_data([0i64, 1i64, 2i64, 3i64], (device, burn::tensor::DType::I64));
        ");
    }

    #[test]
    fn test_onehotencoder_field_init_int_input() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::I64, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe1".to_string(), vec![input], vec![output], config);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let ohe1: Tensor<1, Int> = Tensor::<
            1,
            Int,
        >::from_data([0i64, 1i64, 2i64], (device, burn::tensor::DType::I64));
        ");
    }
}
