use super::prelude::*;

impl NodeCodegen for onnx_ir::split::SplitNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let axis = self.config.axis.to_tokens();

        let outputs = self.outputs.iter().map(arg_to_ident).collect::<Vec<_>>();

        let unpack_outputs = quote! {
            let [#(#outputs),*] = split_tensors.try_into().unwrap();
        };

        // split_with_sizes drops zero-length parts, which would leave fewer tensors than
        // outputs. Sizes that may contain a zero are sliced one part at a time instead.
        let slice_parts = |sizes: TokenStream| {
            quote! {
                let split_tensors = {
                    let mut start = 0;
                    #sizes
                        .into_iter()
                        .map(|size: usize| {
                            let end = start + size;
                            let part = #input.clone().slice_dim(#axis, start..end);
                            start = end;
                            part
                        })
                        .collect::<alloc::vec::Vec<_>>()
                };
                #unpack_outputs
            }
        };

        if let Some(split_sizes_input) = &self.config.split_sizes {
            match split_sizes_input {
                onnx_ir::split::SplitSizesInput::Static(sizes) if !sizes.contains(&0) => {
                    let sizes = sizes.iter().map(|s| s.to_tokens());
                    quote! {
                        let split_tensors = #input.split_with_sizes([#(#sizes),*].into(), #axis);
                        #unpack_outputs
                    }
                }
                onnx_ir::split::SplitSizesInput::Static(sizes) => {
                    let sizes = sizes.iter().map(|s| s.to_tokens());
                    slice_parts(quote! { [#(#sizes),*] })
                }
                // The output count is fixed by the graph; only the sizes wait for run time.
                onnx_ir::split::SplitSizesInput::Runtime(runtime) => {
                    let sizes = scope.arg(&self.inputs[runtime.input_index]);
                    let to_vec = crate::burn::codegen::tensor_to_i64_vec(&sizes);
                    slice_parts(quote! {
                        #to_vec.into_iter().map(|size| size as usize)
                    })
                }
            }
        } else if let Some(split_size) = &self.config.split_size {
            let split_size_tokens = split_size.to_tokens();
            quote! {
                let split_tensors = #input.split(#split_size_tokens, #axis);
                #unpack_outputs
            }
        } else if let Some(num_outputs) = &self.config.num_outputs {
            // Runtime: compute explicit per-output sizes so there are always exactly
            // num_outputs chunks. A short axis leaves trailing parts empty (5 over 4
            // outputs is 2, 2, 1, 0).
            let n = num_outputs.to_tokens();
            let split = slice_parts(quote! { sizes });
            quote! {
                let dim_size = #input.dims()[#axis];
                let chunk = dim_size.div_ceil(#n);
                let sizes: alloc::vec::Vec<usize> = (0..#n)
                    .map(|i| chunk.min(dim_size.saturating_sub(i * chunk)))
                    .collect();
                #split
            }
        } else {
            panic!("Split node must have either split_size, split_sizes, or num_outputs")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::split::{SplitConfig, SplitNodeBuilder, SplitSizesInput};

    #[test]
    fn test_split_equal() {
        let config = SplitConfig {
            axis: 0,
            split_size: Some(2),
            split_sizes: None,
            num_outputs: None,
        };
        let node = SplitNodeBuilder::new("split1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output0", 2, DType::F32)
            .output_tensor("output1", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
            let split_tensors = input.split(2, 0);
            let [output0, output1] = split_tensors.try_into().unwrap();
            (output0, output1)
        }
        ");
    }

    #[test]
    fn test_split_sizes() {
        let config = SplitConfig {
            axis: 1,
            split_size: None,
            split_sizes: Some(SplitSizesInput::Static(vec![1, 3, 2])),
            num_outputs: None,
        };
        let node = SplitNodeBuilder::new("split1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output0", 2, DType::F32)
            .output_tensor("output1", 2, DType::F32)
            .output_tensor("output2", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
            let split_tensors = input.split_with_sizes([1, 3, 2].into(), 1);
            let [output0, output1, output2] = split_tensors.try_into().unwrap();
            (output0, output1, output2)
        }
        ");
    }

    #[test]
    fn test_split_sizes_with_zero() {
        let config = SplitConfig {
            axis: 1,
            split_size: None,
            split_sizes: Some(SplitSizesInput::Static(vec![3, 0, 2])),
            num_outputs: None,
        };
        let node = SplitNodeBuilder::new("split1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output0", 2, DType::F32)
            .output_tensor("output1", 2, DType::F32)
            .output_tensor("output2", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
            let split_tensors = {
                let mut start = 0;
                [3, 0, 2]
                    .into_iter()
                    .map(|size: usize| {
                        let end = start + size;
                        let part = input.clone().slice_dim(1, start..end);
                        start = end;
                        part
                    })
                    .collect::<alloc::vec::Vec<_>>()
            };
            let [output0, output1, output2] = split_tensors.try_into().unwrap();
            (output0, output1, output2)
        }
        ");
    }

    #[test]
    fn test_split_runtime_sizes() {
        let config = SplitConfig {
            axis: 0,
            split_size: None,
            split_sizes: Some(SplitSizesInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "split".to_string(),
                1,
            ))),
            num_outputs: None,
        };
        let node = SplitNodeBuilder::new("split1")
            .input_tensor("input", 1, DType::F32)
            .input_tensor("split", 1, DType::I64)
            .output_tensor("output0", 1, DType::F32)
            .output_tensor("output1", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<1>,
            split: Tensor<1, Int>,
        ) -> (Tensor<1>, Tensor<1>) {
            let split_tensors = {
                let mut start = 0;
                split
                    .to_data()
                    .convert::<i64>()
                    .try_into_vec::<i64>()
                    .unwrap()
                    .into_iter()
                    .map(|size| size as usize)
                    .into_iter()
                    .map(|size: usize| {
                        let end = start + size;
                        let part = input.clone().slice_dim(0, start..end);
                        start = end;
                        part
                    })
                    .collect::<alloc::vec::Vec<_>>()
            };
            let [output0, output1] = split_tensors.try_into().unwrap();
            (output0, output1)
        }
        ");
    }

    #[test]
    fn test_split_runtime_num_outputs() {
        // Test runtime split size calculation using num_outputs
        let config = SplitConfig {
            axis: 2,
            split_size: None,
            split_sizes: None,
            num_outputs: Some(3),
        };
        let node = SplitNodeBuilder::new("split1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output0", 3, DType::F32)
            .output_tensor("output1", 3, DType::F32)
            .output_tensor("output2", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
            let dim_size = input.dims()[2];
            let chunk = dim_size.div_ceil(3);
            let sizes: alloc::vec::Vec<usize> = (0..3)
                .map(|i| chunk.min(dim_size.saturating_sub(i * chunk)))
                .collect();
            let split_tensors = {
                let mut start = 0;
                sizes
                    .into_iter()
                    .map(|size: usize| {
                        let end = start + size;
                        let part = input.clone().slice_dim(2, start..end);
                        start = end;
                        part
                    })
                    .collect::<alloc::vec::Vec<_>>()
            };
            let [output0, output1, output2] = split_tensors.try_into().unwrap();
            (output0, output1, output2)
        }
        ");
    }
}
