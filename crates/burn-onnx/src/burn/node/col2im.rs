use super::prelude::*;

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

        let image_shape = &self.config.image_shape;
        let block_shape = &self.config.block_shape;
        let strides = &self.config.strides;
        let dilations = &self.config.dilations;
        let pads = &self.config.pads;

        let num_spatial_dims = image_shape.len();

        // Pre-compute begin and end pads for each spatial dimension
        let pads_begin: Vec<usize> = pads[..num_spatial_dims].to_vec();
        let pads_end: Vec<usize> = pads[num_spatial_dims..].to_vec();

        // Compute effective block sizes with dilation: d_i * (block_i - 1) + 1
        let effective_blocks: Vec<usize> = block_shape
            .iter()
            .zip(dilations.iter())
            .map(|(&b, &d)| d * (b - 1) + 1)
            .collect();

        // Compute number of output positions (L_i) per spatial dimension
        // L_i = (image_shape_i + pad_begin_i + pad_end_i - effective_block_i) / stride_i + 1
        let output_counts: Vec<usize> = (0..num_spatial_dims)
            .map(|i| {
                (image_shape[i] + pads_begin[i] + pads_end[i] - effective_blocks[i]) / strides[i]
                    + 1
            })
            .collect();

        // Total number of sliding window positions
        let total_positions: usize = output_counts.iter().product();

        // Product of block_shape (number of elements per block)
        let block_product: usize = block_shape.iter().product();

        // For 2D case, generate optimized code
        if num_spatial_dims == 2 {
            let img_h = image_shape[0];
            let img_w = image_shape[1];
            let block_h = block_shape[0];
            let block_w = block_shape[1];
            let stride_h = strides[0];
            let stride_w = strides[1];
            let dilation_h = dilations[0];
            let dilation_w = dilations[1];
            let pad_h_begin = pads_begin[0];
            let pad_w_begin = pads_begin[1];
            let out_w = output_counts[1];

            let pad_h_end = pads_end[0];
            let pad_w_end = pads_end[1];

            let check_body = if pad_h_begin == 0
                && pad_w_begin == 0
                && pad_h_end == 0
                && pad_w_end == 0
            {
                // Optimized check for zero padding
                quote! {
                    if h_idx < #img_h && w_idx < #img_w {
                        let h_out = h_idx;
                        let w_out = w_idx;

                        // Extract element and add to result
                        let val = col_slice
                            .clone()
                            .slice([0..batch_size, 0..channels, bh..bh + 1, bw..bw + 1]);
                        let current = result
                            .clone()
                            .slice([0..batch_size, 0..channels, h_out..h_out + 1, w_out..w_out + 1]);
                        result = result.slice_assign(
                            [0..batch_size, 0..channels, h_out..h_out + 1, w_out..w_out + 1],
                            current + val,
                        );
                    }
                }
            } else {
                // General check for non-zero padding, optimized per dimension
                let h_lower_check = if pad_h_begin > 0 {
                    quote! { h_idx >= #pad_h_begin && }
                } else {
                    quote! {}
                };

                let h_upper_check = if pad_h_end > 0 {
                    quote! { h_idx < #img_h + #pad_h_begin }
                } else {
                    quote! { h_idx < #img_h }
                };

                let w_lower_check = if pad_w_begin > 0 {
                    quote! { && w_idx >= #pad_w_begin && }
                } else {
                    quote! { && }
                };

                let w_upper_check = if pad_w_end > 0 {
                    quote! { w_idx < #img_w + #pad_w_begin }
                } else {
                    quote! { w_idx < #img_w }
                };

                let h_out_assign = if pad_h_begin > 0 {
                    quote! { let h_out = h_idx - #pad_h_begin; }
                } else {
                    quote! { let h_out = h_idx; }
                };

                let w_out_assign = if pad_w_begin > 0 {
                    quote! { let w_out = w_idx - #pad_w_begin; }
                } else {
                    quote! { let w_out = w_idx; }
                };

                quote! {
                    if #h_lower_check #h_upper_check #w_lower_check #w_upper_check
                    {
                        #h_out_assign
                        #w_out_assign

                        // Extract element and add to result
                        let val = col_slice
                            .clone()
                            .slice([0..batch_size, 0..channels, bh..bh + 1, bw..bw + 1]);
                        let current = result
                            .clone()
                            .slice([0..batch_size, 0..channels, h_out..h_out + 1, w_out..w_out + 1]);
                        result = result.slice_assign(
                            [0..batch_size, 0..channels, h_out..h_out + 1, w_out..w_out + 1],
                            current + val,
                        );
                    }
                }
            };

            quote! {
                let #output = {
                    let [batch_size, col_channels, _l] = #input.shape().dims();
                    let channels = col_channels / #block_product;
                    let device = #input.device();

                    // Create output tensor with zeros: [N, C, H, W]
                    let mut result = Tensor::<B, 4>::zeros([batch_size, channels, #img_h, #img_w], &device);

                    // Reshape input to [N, C, block_h * block_w, L]
                    let input_reshaped = #input.reshape([batch_size, channels, #block_product, #total_positions]);

                    // Iterate over sliding window positions and accumulate
                    for pos in 0..#total_positions {
                        let pos_h = pos / #out_w;
                        let pos_w = pos % #out_w;

                        // Extract the column for this position: [N, C, block_h * block_w]
                        let col_slice = input_reshaped
                            .clone()
                            .slice([0..batch_size, 0..channels, 0..#block_product, pos..pos + 1])
                            .reshape([batch_size, channels, #block_h, #block_w]);

                        // Scatter into output at the correct positions
                        for bh in 0..#block_h {
                            for bw in 0..#block_w {
                                let h_idx = pos_h * #stride_h + bh * #dilation_h;
                                let w_idx = pos_w * #stride_w + bw * #dilation_w;

                                // Check padding boundaries
                                #check_body
                            }
                        }
                    }

                    result
                };
            }
        } else if num_spatial_dims == 1 {
            // For 1D case
            let img_len = image_shape[0];
            let block_len = block_shape[0];
            let stride_len = strides[0];
            let dilation_len = dilations[0];
            let pad_begin = pads_begin[0];

            let check_body = if pad_begin == 0 {
                // Optimized check for zero padding
                quote! {
                    if idx < #img_len {
                        let out_idx = idx;

                        let val = col_slice
                            .clone()
                            .slice([0..batch_size, 0..channels, bi..bi + 1]);
                        let current = result
                            .clone()
                            .slice([0..batch_size, 0..channels, out_idx..out_idx + 1]);
                        result = result.slice_assign(
                            [0..batch_size, 0..channels, out_idx..out_idx + 1],
                            current + val,
                        );
                    }
                }
            } else {
                // General check for non-zero padding, optimized
                let lower_check = if pad_begin > 0 {
                    quote! { idx >= #pad_begin && }
                } else {
                    quote! {}
                };

                let pad_end = pads_end[0];
                let upper_check = if pad_end > 0 {
                    quote! { idx < #img_len + #pad_begin }
                } else {
                    quote! { idx < #img_len }
                };

                let out_idx_assign = if pad_begin > 0 {
                    quote! { let out_idx = idx - #pad_begin; }
                } else {
                    quote! { let out_idx = idx; }
                };

                quote! {
                    if #lower_check #upper_check {
                        #out_idx_assign

                        let val = col_slice
                            .clone()
                            .slice([0..batch_size, 0..channels, bi..bi + 1]);
                        let current = result
                            .clone()
                            .slice([0..batch_size, 0..channels, out_idx..out_idx + 1]);
                        result = result.slice_assign(
                            [0..batch_size, 0..channels, out_idx..out_idx + 1],
                            current + val,
                        );
                    }
                }
            };

            quote! {
                let #output = {
                    let [batch_size, col_channels, _l] = #input.shape().dims();
                    let channels = col_channels / #block_product;
                    let device = #input.device();

                    // Create output tensor with zeros: [N, C, L]
                    let mut result = Tensor::<B, 3>::zeros([batch_size, channels, #img_len], &device);

                    // Reshape input to [N, C, block_len, num_positions]
                    let input_reshaped = #input.reshape([batch_size, channels, #block_product, #total_positions]);

                    // Iterate over sliding window positions and accumulate
                    for pos in 0..#total_positions {
                        let col_slice = input_reshaped
                            .clone()
                            .slice([0..batch_size, 0..channels, 0..#block_product, pos..pos + 1])
                            .reshape([batch_size, channels, #block_len]);

                        for bi in 0..#block_len {
                            let idx = pos * #stride_len + bi * #dilation_len;

                            #check_body
                        }
                    }

                    result
                };
            }
        } else {
            // Panic for unsupported dimensions > 2D
            panic!(
                "Col2Im only supports 1D and 2D spatial dimensions, got {}",
                num_spatial_dims
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::col2im::{Col2ImConfig, Col2ImNodeBuilder};

    #[test]
    fn test_col2im_2d_basic() {
        let config = Col2ImConfig::new(
            vec![5, 5],       // image_shape
            vec![2, 2],       // block_shape
            vec![1, 1],       // dilations
            vec![0, 0, 0, 0], // pads
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }

    #[test]
    fn test_col2im_2d_with_strides() {
        let config = Col2ImConfig::new(
            vec![6, 6],       // image_shape
            vec![2, 2],       // block_shape
            vec![1, 1],       // dilations
            vec![0, 0, 0, 0], // pads
            vec![2, 2],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im2")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }

    #[test]
    fn test_col2im_2d_with_padding() {
        let config = Col2ImConfig::new(
            vec![5, 5],       // image_shape
            vec![2, 2],       // block_shape
            vec![1, 1],       // dilations
            vec![1, 1, 1, 1], // pads [t, l, b, r]
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_pad")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }

    #[test]
    fn test_col2im_2d_with_dilation() {
        let config = Col2ImConfig::new(
            vec![5, 5],       // image_shape
            vec![2, 2],       // block_shape
            vec![2, 2],       // dilations
            vec![0, 0, 0, 0], // pads
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_dil")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }

    #[test]
    fn test_col2im_1d_basic() {
        let config = Col2ImConfig::new(
            vec![10],   // image_shape
            vec![3],    // block_shape
            vec![1],    // dilations
            vec![0, 0], // pads
            vec![1],    // strides
        );
        let node = Col2ImNodeBuilder::new("col2im3")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }
    #[test]
    fn test_col2im_1d_with_padding() {
        let config = Col2ImConfig::new(
            vec![10],   // image_shape
            vec![3],    // block_shape
            vec![1],    // dilations
            vec![1, 1], // pads [begin, end]
            vec![1],    // strides
        );
        let node = Col2ImNodeBuilder::new("col2im4")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }
    #[test]
    fn test_col2im_2d_with_end_padding() {
        let config = Col2ImConfig::new(
            vec![5, 5],       // image_shape
            vec![2, 2],       // block_shape
            vec![1, 1],       // dilations
            vec![0, 0, 1, 1], // pads [t, l, b, r] - only end padding
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_end_pad")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }
}
