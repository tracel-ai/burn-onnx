use super::prelude::*;
use onnx_ir::col2im::Col2ImShape;

impl NodeCodegen for onnx_ir::col2im::Col2ImNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let config = &self.config;
        let num_spatial_dims = config.image_shape.len();

        // fold4d is 2D. A 1D Col2Im is the same fold with a height-1 kernel and image.
        let lift = |values: &[usize], fill: usize| -> [usize; 2] {
            match values {
                [w] => [fill, *w],
                [h, w] => [*h, *w],
                _ => unreachable!("Col2Im spatial rank validated in onnx-ir"),
            }
        };
        let strides = lift(&config.strides, 1);
        let dilations = lift(&config.dilations, 1);
        let pads_begin = lift(&config.pads[..num_spatial_dims], 0);
        let pads_end = lift(&config.pads[num_spatial_dims..], 0);

        // fold4d pads both sides equally. Asymmetric pads fold onto the padded canvas
        // with no padding and crop the image out of it.
        let symmetric = pads_begin == pads_end;
        let fold_padding = if symmetric { pads_begin } else { [0, 0] };
        let [sh, sw] = strides;
        let [ph, pw] = fold_padding;
        let [dh, dw] = dilations;
        let options = quote! {
            burn::tensor::ops::UnfoldOptions::new([#sh, #sw], [#ph, #pw], [#dh, #dw])
        };
        let [top, left] = pads_begin;
        let [pad_h, pad_w] = [pads_begin[0] + pads_end[0], pads_begin[1] + pads_end[1]];

        let reshape_1d = |folded: TokenStream| {
            if num_spatial_dims == 1 {
                quote! {{
                    let folded: Tensor<4> = #folded;
                    let [batch, channels, _, width] = folded.dims();
                    folded.reshape([batch, channels, width])
                }}
            } else {
                folded
            }
        };

        // Static shapes become literals, so both kinds share one body.
        let mut shape_tokens = |shape: &Col2ImShape| -> TokenStream {
            let values = match shape {
                Col2ImShape::Static(values) => {
                    let [a, b] = lift(values, 1);
                    return quote! { [#a, #b] };
                }
                Col2ImShape::Runtime { input, .. } => {
                    let arg = &self.inputs[input.input_index];
                    let value = scope.arg(arg);
                    match &arg.ty {
                        ArgType::Shape(_) => quote! { #value },
                        _ => crate::burn::codegen::tensor_to_i64_vec(&value),
                    }
                }
            };
            if num_spatial_dims == 1 {
                quote! {{ let values = #values; [1, values[0] as usize] }}
            } else {
                quote! {{ let values = #values; [values[0] as usize, values[1] as usize] }}
            }
        };
        let image = shape_tokens(&config.image_shape);
        let kernel = shape_tokens(&config.block_shape);
        let fold_size = if symmetric {
            quote! { image }
        } else {
            quote! { [image[0] + #pad_h, image[1] + #pad_w] }
        };
        let fold = quote! {
            burn::tensor::module::fold4d(input, #fold_size, kernel, #options)
        };
        let folded = if symmetric {
            fold
        } else {
            quote! { #fold.slice(s![.., .., #top..#top + image[0], #left..#left + image[1]]) }
        };
        let folded = reshape_1d(folded);
        // One statement reads every graph value, so the locals cannot shadow one.
        quote! {
            let #output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) =
                    (#input, #image, #kernel);
                #folded
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::col2im::{Col2ImConfig, Col2ImNodeBuilder, Col2ImShape};

    #[test]
    fn test_col2im_2d_basic() {
        let config = Col2ImConfig::new(
            Col2ImShape::Static(vec![5, 5]), // image_shape
            Col2ImShape::Static(vec![2, 2]), // block_shape
            vec![1, 1],                      // dilations
            vec![0, 0, 0, 0],                // pads
            vec![1, 1],                      // strides
        );
        let node = Col2ImNodeBuilder::new("col2im1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<4> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    [5usize, 5usize],
                    [2usize, 2usize],
                );
                burn::tensor::module::fold4d(
                    input,
                    image,
                    kernel,
                    burn::tensor::ops::UnfoldOptions::new(
                        [1usize, 1usize],
                        [0usize, 0usize],
                        [1usize, 1usize],
                    ),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_col2im_2d_with_padding() {
        let config = Col2ImConfig::new(
            Col2ImShape::Static(vec![5, 5]), // image_shape
            Col2ImShape::Static(vec![2, 2]), // block_shape
            vec![1, 1],                      // dilations
            vec![1, 1, 1, 1],                // pads [t, l, b, r]
            vec![1, 1],                      // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_pad")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<4> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    [5usize, 5usize],
                    [2usize, 2usize],
                );
                burn::tensor::module::fold4d(
                    input,
                    image,
                    kernel,
                    burn::tensor::ops::UnfoldOptions::new(
                        [1usize, 1usize],
                        [1usize, 1usize],
                        [1usize, 1usize],
                    ),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_col2im_2d_with_asymmetric_padding() {
        let config = Col2ImConfig::new(
            Col2ImShape::Static(vec![5, 5]), // image_shape
            Col2ImShape::Static(vec![2, 2]), // block_shape
            vec![1, 1],                      // dilations
            vec![0, 1, 1, 0],                // pads [t, l, b, r]
            vec![1, 1],                      // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_pad")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<4> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    [5usize, 5usize],
                    [2usize, 2usize],
                );
                burn::tensor::module::fold4d(
                        input,
                        [image[0] + 1usize, image[1] + 1usize],
                        kernel,
                        burn::tensor::ops::UnfoldOptions::new(
                            [1usize, 1usize],
                            [0usize, 0usize],
                            [1usize, 1usize],
                        ),
                    )
                    .slice(s![.., .., 0usize..0usize + image[0], 1usize..1usize + image[1]])
            };
            output
        }
        ");
    }

    #[test]
    fn test_col2im_2d_with_strides() {
        let config = Col2ImConfig::new(
            Col2ImShape::Static(vec![6, 6]), // image_shape
            Col2ImShape::Static(vec![2, 2]), // block_shape
            vec![1, 1],                      // dilations
            vec![0, 0, 0, 0],                // pads
            vec![2, 2],                      // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_stride")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<4> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    [6usize, 6usize],
                    [2usize, 2usize],
                );
                burn::tensor::module::fold4d(
                    input,
                    image,
                    kernel,
                    burn::tensor::ops::UnfoldOptions::new(
                        [2usize, 2usize],
                        [0usize, 0usize],
                        [1usize, 1usize],
                    ),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_col2im_2d_with_dilation() {
        let config = Col2ImConfig::new(
            Col2ImShape::Static(vec![5, 5]), // image_shape
            Col2ImShape::Static(vec![2, 2]), // block_shape
            vec![2, 2],                      // dilations
            vec![0, 0, 0, 0],                // pads
            vec![1, 1],                      // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_dil")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<4> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    [5usize, 5usize],
                    [2usize, 2usize],
                );
                burn::tensor::module::fold4d(
                    input,
                    image,
                    kernel,
                    burn::tensor::ops::UnfoldOptions::new(
                        [1usize, 1usize],
                        [0usize, 0usize],
                        [2usize, 2usize],
                    ),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_col2im_1d_basic() {
        let config = Col2ImConfig::new(
            Col2ImShape::Static(vec![10]), // image_shape
            Col2ImShape::Static(vec![3]),  // block_shape
            vec![1],                       // dilations
            vec![0, 0],                    // pads
            vec![1],                       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im1d")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    [1usize, 10usize],
                    [1usize, 3usize],
                );
                {
                    let folded: Tensor<4> = burn::tensor::module::fold4d(
                        input,
                        image,
                        kernel,
                        burn::tensor::ops::UnfoldOptions::new(
                            [1usize, 1usize],
                            [0usize, 0usize],
                            [1usize, 1usize],
                        ),
                    );
                    let [batch, channels, _, width] = folded.dims();
                    folded.reshape([batch, channels, width])
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_col2im_2d_runtime_shapes() {
        let config = Col2ImConfig::new(
            Col2ImShape::Runtime {
                input: onnx_ir::ir::RuntimeInputRef::new("image_shape".to_string(), 1),
                len: 2,
            },
            Col2ImShape::Runtime {
                input: onnx_ir::ir::RuntimeInputRef::new("block_shape".to_string(), 2),
                len: 2,
            },
            vec![1, 1],       // dilations
            vec![0, 1, 0, 1], // pads [t, l, b, r]
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_runtime")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("image_shape", 1, DType::I64)
            .input_tensor("block_shape", 1, DType::I64)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            image_shape: Tensor<1, Int>,
            block_shape: Tensor<1, Int>,
        ) -> Tensor<4> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    {
                        let values = image_shape
                            .to_data()
                            .convert::<i64>()
                            .try_into_vec::<i64>()
                            .unwrap();
                        [values[0] as usize, values[1] as usize]
                    },
                    {
                        let values = block_shape
                            .to_data()
                            .convert::<i64>()
                            .try_into_vec::<i64>()
                            .unwrap();
                        [values[0] as usize, values[1] as usize]
                    },
                );
                burn::tensor::module::fold4d(
                    input,
                    image,
                    kernel,
                    burn::tensor::ops::UnfoldOptions::new(
                        [1usize, 1usize],
                        [0usize, 1usize],
                        [1usize, 1usize],
                    ),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_col2im_1d_runtime_image_shape() {
        let config = Col2ImConfig::new(
            Col2ImShape::Runtime {
                input: onnx_ir::ir::RuntimeInputRef::new("image_shape".to_string(), 1),
                len: 1,
            },
            Col2ImShape::Static(vec![3]),
            vec![1],    // dilations
            vec![0, 0], // pads
            vec![1],    // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_runtime")
            .input_tensor("input", 3, DType::F32)
            .input_shape("image_shape", 1)
            .input_tensor("block_shape", 1, DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            image_shape: [i64; 1],
            block_shape: Tensor<1, Int>,
        ) -> Tensor<3> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    {
                        let values = image_shape;
                        [1, values[0] as usize]
                    },
                    [1usize, 3usize],
                );
                {
                    let folded: Tensor<4> = burn::tensor::module::fold4d(
                        input,
                        image,
                        kernel,
                        burn::tensor::ops::UnfoldOptions::new(
                            [1usize, 1usize],
                            [0usize, 0usize],
                            [1usize, 1usize],
                        ),
                    );
                    let [batch, channels, _, width] = folded.dims();
                    folded.reshape([batch, channels, width])
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_col2im_2d_runtime_asymmetric_padding() {
        let config = Col2ImConfig::new(
            Col2ImShape::Runtime {
                input: onnx_ir::ir::RuntimeInputRef::new("image_shape".to_string(), 1),
                len: 2,
            },
            Col2ImShape::Static(vec![2, 2]),
            vec![1, 1],       // dilations
            vec![1, 0, 0, 2], // pads [t, l, b, r]
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_runtime")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("image_shape", 1, DType::I64)
            .input_tensor("block_shape", 1, DType::I64)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            image_shape: Tensor<1, Int>,
            block_shape: Tensor<1, Int>,
        ) -> Tensor<4> {
            let output = {
                let (input, image, kernel): (_, [usize; 2], [usize; 2]) = (
                    input,
                    {
                        let values = image_shape
                            .to_data()
                            .convert::<i64>()
                            .try_into_vec::<i64>()
                            .unwrap();
                        [values[0] as usize, values[1] as usize]
                    },
                    [2usize, 2usize],
                );
                burn::tensor::module::fold4d(
                        input,
                        [image[0] + 1usize, image[1] + 2usize],
                        kernel,
                        burn::tensor::ops::UnfoldOptions::new(
                            [1usize, 1usize],
                            [0usize, 0usize],
                            [1usize, 1usize],
                        ),
                    )
                    .slice(s![.., .., 1usize..1usize + image[0], 0usize..0usize + image[1]])
            };
            output
        }
        ");
    }
}
