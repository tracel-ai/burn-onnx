use super::prelude::*;

impl NodeCodegen for onnx_ir::node::argmax::ArgMaxNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let axis_usize = self.config.axis;
        let axis = axis_usize.to_tokens();
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        // When ONNX `select_last_index=1`, return the index of the last
        // occurrence of the maximum along the axis. Burn's argmax returns
        // the first occurrence, so we rewrite as
        //   axis_size - 1 - argmax(flip(input, axis), axis)
        // For ties at positions i1 < i2 < ... < ik in the original tensor,
        // the flipped tensor has those same values at positions
        // axis_size-1-ik < ... < axis_size-1-i1, so Burn's first-occurrence
        // argmax on the flipped tensor lands on axis_size-1-ik; subtracting
        // from axis_size-1 recovers ik, the last tied index in the
        // original, matching ONNX `select_last_index=1` semantics.
        let argmax_expr = if self.config.select_last_index {
            let axis_isize = axis_usize as isize;
            quote! {
                {
                    let __argmax_input = #input;
                    let __argmax_axis_size =
                        __argmax_input.shape()[#axis_usize] as i64;
                    __argmax_input
                        .flip([#axis_isize])
                        .argmax(#axis)
                        .mul_scalar(-1i64)
                        .add_scalar(__argmax_axis_size - 1)
                }
            }
        } else {
            quote! { #input.argmax(#axis) }
        };

        // argmax uses the backend's default int element type, which varies across backends.
        // Cast to the ONNX-specified dtype to avoid DTypeMismatch on backends whose default differs.
        match &output_arg.ty {
            onnx_ir::ir::ArgType::Tensor(tensor) => {
                let output_dtype = tensor.dtype.to_tokens();
                if self.config.keepdims {
                    quote! {
                        let #output = #argmax_expr.cast(#output_dtype);
                    }
                } else {
                    let output_rank = tensor.rank;
                    quote! {
                        let argmax_result = #argmax_expr;
                        let #output = argmax_result.squeeze_dim::<#output_rank>(#axis).cast(#output_dtype);
                    }
                }
            }
            onnx_ir::ir::ArgType::ScalarTensor(dtype) => {
                let output_dtype = dtype.to_tokens();
                quote! {
                    let #output = #argmax_expr.reshape([1]).cast(#output_dtype);
                }
            }
            onnx_ir::ir::ArgType::ScalarNative(_) => {
                // Extracted to native scalar, no cast needed
                quote! {
                    let argmax_result = #argmax_expr;
                    let #output = argmax_result.into_scalar().elem::<i64>();
                }
            }
            _ => panic!("ArgMax output must be Tensor or Scalar"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::argmax::{ArgMaxConfig, ArgMaxNodeBuilder};

    #[test]
    fn test_argmax_keepdims() {
        let config = ArgMaxConfig::new(1, true, false);
        let node = ArgMaxNodeBuilder::new("argmax1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3, Int> {
            let output = input.argmax(1).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmax_no_keepdims() {
        let config = ArgMaxConfig::new(2, false, false);
        let node = ArgMaxNodeBuilder::new("argmax2")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3, Int> {
            let argmax_result = input.argmax(2);
            let output = argmax_result.squeeze_dim::<3usize>(2).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmax_scalar_tensor_output() {
        let config = ArgMaxConfig::new(0, false, false);
        let node = ArgMaxNodeBuilder::new("argmax3")
            .input_tensor("input", 1, DType::F32)
            .output_scalar_tensor("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1, Int> {
            let output = input.argmax(0).reshape([1]).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmax_scalar_native_output() {
        let config = ArgMaxConfig::new(0, false, false);
        let node = ArgMaxNodeBuilder::new("argmax4")
            .input_tensor("input", 1, DType::F32)
            .output_scalar("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> i64 {
            let argmax_result = input.argmax(0);
            let output = argmax_result.into_scalar().elem::<i64>();
            output
        }
        ");
    }

    #[test]
    fn test_argmax_select_last_index_keepdims() {
        // select_last_index=1 rewrites argmax(input, axis) as
        //     axis_size - 1 - argmax(flip(input, axis), axis)
        let config = ArgMaxConfig::new(1, true, true);
        let node = ArgMaxNodeBuilder::new("argmax_last")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3, Int> {
            let output = {
                let __argmax_input = input;
                let __argmax_axis_size = __argmax_input.shape()[1usize] as i64;
                __argmax_input
                    .flip([1isize])
                    .argmax(1)
                    .mul_scalar(-1i64)
                    .add_scalar(__argmax_axis_size - 1)
            }
                .cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmax_select_last_index_no_keepdims() {
        let config = ArgMaxConfig::new(2, false, true);
        let node = ArgMaxNodeBuilder::new("argmax_last_no_keep")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3, Int> {
            let argmax_result = {
                let __argmax_input = input;
                let __argmax_axis_size = __argmax_input.shape()[2usize] as i64;
                __argmax_input
                    .flip([2isize])
                    .argmax(2)
                    .mul_scalar(-1i64)
                    .add_scalar(__argmax_axis_size - 1)
            };
            let output = argmax_result.squeeze_dim::<3usize>(2).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }
}
