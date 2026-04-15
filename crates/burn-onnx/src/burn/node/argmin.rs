use super::prelude::*;

impl NodeCodegen for onnx_ir::node::argmin::ArgMinNode {
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
        // occurrence of the minimum along the axis. Burn's argmin returns
        // the first occurrence, so we rewrite as
        //   axis_size - 1 - argmin(flip(input, axis), axis)
        // The same reasoning as argmax applies: ties at positions
        // i1 < ... < ik in the original land at axis_size-1-ik ... in the
        // flipped tensor, so Burn's first-occurrence argmin there gives
        // axis_size-1-ik, and the subtraction recovers ik (the last tied
        // index), matching ONNX `select_last_index=1` semantics.
        let argmin_expr = if self.config.select_last_index {
            let axis_isize = axis_usize as isize;
            quote! {
                {
                    let __argmin_input = #input;
                    let __argmin_axis_size =
                        __argmin_input.shape()[#axis_usize] as i64;
                    __argmin_input
                        .flip([#axis_isize])
                        .argmin(#axis)
                        .mul_scalar(-1i64)
                        .add_scalar(__argmin_axis_size - 1)
                }
            }
        } else {
            quote! { #input.argmin(#axis) }
        };

        // argmin uses the backend's default int element type, which varies across backends.
        // Cast to the ONNX-specified dtype to avoid DTypeMismatch on backends whose default differs.
        match &output_arg.ty {
            onnx_ir::ir::ArgType::Tensor(tensor) => {
                let output_dtype = tensor.dtype.to_tokens();
                if self.config.keepdims {
                    quote! {
                        let #output = #argmin_expr.cast(#output_dtype);
                    }
                } else {
                    let output_rank = tensor.rank;
                    quote! {
                        let argmin_result = #argmin_expr;
                        let #output = argmin_result.squeeze_dim::<#output_rank>(#axis).cast(#output_dtype);
                    }
                }
            }
            onnx_ir::ir::ArgType::ScalarTensor(dtype) => {
                let output_dtype = dtype.to_tokens();
                quote! {
                    let #output = #argmin_expr.reshape([1]).cast(#output_dtype);
                }
            }
            onnx_ir::ir::ArgType::ScalarNative(_) => {
                // Extracted to native scalar, no cast needed
                quote! {
                    let argmin_result = #argmin_expr;
                    let #output = argmin_result.into_scalar().elem::<i64>();
                }
            }
            _ => panic!("ArgMin output must be Tensor or Scalar"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::argmin::{ArgMinConfig, ArgMinNodeBuilder};

    #[test]
    fn test_argmin_keepdims() {
        let config = ArgMinConfig::new(1, true, false);
        let node = ArgMinNodeBuilder::new("argmin1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3, Int> {
            let output = input.argmin(1).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmin_no_keepdims() {
        let config = ArgMinConfig::new(0, false, false);
        let node = ArgMinNodeBuilder::new("argmin2")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1, Int> {
            let argmin_result = input.argmin(0);
            let output = argmin_result.squeeze_dim::<1usize>(0).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmin_scalar_tensor_output() {
        let config = ArgMinConfig::new(0, false, false);
        let node = ArgMinNodeBuilder::new("argmin3")
            .input_tensor("input", 1, DType::F32)
            .output_scalar_tensor("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1, Int> {
            let output = input.argmin(0).reshape([1]).cast(burn::tensor::DType::I64);
            output
        }
        ");
    }

    #[test]
    fn test_argmin_scalar_native_output() {
        let config = ArgMinConfig::new(0, false, false);
        let node = ArgMinNodeBuilder::new("argmin4")
            .input_tensor("input", 1, DType::F32)
            .output_scalar("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> i64 {
            let argmin_result = input.argmin(0);
            let output = argmin_result.into_scalar().elem::<i64>();
            output
        }
        ");
    }

    #[test]
    fn test_argmin_select_last_index_keepdims() {
        let config = ArgMinConfig::new(1, true, true);
        let node = ArgMinNodeBuilder::new("argmin_last")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3, Int> {
            let output = {
                let __argmin_input = input;
                let __argmin_axis_size = __argmin_input.shape()[1usize] as i64;
                __argmin_input
                    .flip([1isize])
                    .argmin(1)
                    .mul_scalar(-1i64)
                    .add_scalar(__argmin_axis_size - 1)
            }
                .cast(burn::tensor::DType::I64);
            output
        }
        ");
    }
}
