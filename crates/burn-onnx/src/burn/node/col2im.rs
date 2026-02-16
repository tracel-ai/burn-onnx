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

        // Split pads into begin/end
        let pads_begin: Vec<usize> = pads[..num_spatial_dims].to_vec();
        let pads_end: Vec<usize> = pads[num_spatial_dims..].to_vec();

        // Compute effective block sizes: d * (k - 1) + 1
        let effective_blocks: Vec<usize> = block_shape
            .iter()
            .zip(dilations.iter())
            .map(|(&b, &d)| d * (b - 1) + 1)
            .collect();

        // Compute output windows count per dimension
        // L_i = (img + pad_begin + pad_end - effective) / stride + 1
        let output_counts: Vec<usize> = (0..num_spatial_dims)
            .map(|i| {
                (image_shape[i] + pads_begin[i] + pads_end[i] - effective_blocks[i]) / strides[i]
                    + 1
            })
            .collect();

        let total_windows: usize = output_counts.iter().product();
        let block_product: usize = block_shape.iter().product();
        let total_input_elements = block_product * total_windows;

        // Compute Padded Output Shape dimensions
        let padded_dims: Vec<usize> = (0..num_spatial_dims)
            .map(|i| image_shape[i] + pads_begin[i] + pads_end[i])
            .collect();

        // Calculate the linear indices for scatter-add
        // We compute where each element of the input (flattened block * windows) goes in the flattened padded output.
        // Input layout: [Batch, Channel, BlockElements, Windows] -> Flattened last 2 dims: [BlockElements, Windows]
        // But Burn reshape/flatten is usually C-order (row-major).
        // Input tensor is [N, C, BlockProd, L] -> reshape to [N, C, BlockProd * L]
        // So we iterate: for window in 0..windows { for block_elem in 0..block_product { ... } } (?)
        // Wait, reshape [N, C, Block, L] -> [N, C, Block*L] means inner dimension is L.
        // So iterate block_elem outer, window inner? No, Burn/Numpy default layout is standard (last dim contiguous).
        // So [d0, d1, d2, d3] -> [d0, d1, d2*d3] means d3 is the fastest changing index.
        // So index = block_idx * total_windows + window_idx

        // BUT, Col2Im input is usually [N, C*BlockProd, L] in ONNX spec.
        // My onnx-ir says: "Input data tensor from Im2Col, shape [N, C * product(block_shape), L]"
        // So we interpret it as [N, C, BlockProd, L] for reshaping purposes (logic in type inference confirms this).
        // So element at index `i` in flattened spatial dim corresponds to:
        //   block_idx = i / total_windows
        //   window_idx = i % total_windows

        // Prepare indices computation tokens
        let padded_size: usize = padded_dims.iter().product();
        let mut stride_acc = 1;
        let mut block_terms = Vec::new();
        let mut window_terms = Vec::new();

        for i in (0..num_spatial_dims).rev() {
            let dim_stride_acc = stride_acc;
            stride_acc *= padded_dims[i];

            // Block Term for dim i
            let b_size = block_shape[i];
            let b_dilation_stride = (dilations[i] * dim_stride_acc) as i64;

            let mut b_shape_dims = vec![1usize; num_spatial_dims];
            b_shape_dims[i] = b_size;
            let b_shape_tokens = quote! { [#(#b_shape_dims),*] };

            block_terms.push(quote! {
                Tensor::<B, 1, Int>::arange(0..#b_size as i64, &device)
                    .mul_scalar(#b_dilation_stride)
                    .reshape(#b_shape_tokens)
            });

            // Window Term for dim i
            let w_size = output_counts[i];
            let w_stride_stride = (strides[i] * dim_stride_acc) as i64;

            let mut w_shape_dims = vec![1usize; num_spatial_dims];
            w_shape_dims[i] = w_size;
            let w_shape_tokens = quote! { [#(#w_shape_dims),*] };

            window_terms.push(quote! {
                Tensor::<B, 1, Int>::arange(0..#w_size as i64, &device)
                    .mul_scalar(#w_stride_stride)
                    .reshape(#w_shape_tokens)
            });
        }

        let block_sum = if block_terms.is_empty() {
            quote! { Tensor::<B, 1, Int>::zeros([1], &device) }
        } else {
            let first = &block_terms[0];
            let rest = &block_terms[1..];
            if rest.is_empty() {
                quote! { #first }
            } else {
                quote! { #first #( + #rest)* }
            }
        };

        let window_sum = if window_terms.is_empty() {
            quote! { Tensor::<B, 1, Int>::zeros([1], &device) }
        } else {
            let first = &window_terms[0];
            let rest = &window_terms[1..];
            if rest.is_empty() {
                quote! { #first }
            } else {
                quote! { #first #( + #rest)* }
            }
        };

        let indices_computation_code = quote! {
             let block_offsets = (#block_sum).reshape([#block_product, 1]);
             let window_offsets = (#window_sum).reshape([1, #total_windows]);
             (block_offsets + window_offsets).reshape([-1])
        };

        // Padded shape for reshape
        let padded_shape_tokens = match num_spatial_dims {
            1 => quote! { [batch_size, channels, #padded_size] },
            2 => {
                let h_pad = padded_dims[0];
                let w_pad = padded_dims[1];
                quote! { [batch_size, channels, #h_pad, #w_pad] }
            }
            _ => unreachable!("Unsupported dimensions checked by infer_types"),
        };

        let has_padding = pads_begin.iter().any(|&p| p != 0) || pads_end.iter().any(|&p| p != 0);

        let slice_logic = if !has_padding {
            // Optimization: If no padding, just return the reshaped canvas.
            quote! { canvas.reshape(#padded_shape_tokens) }
        } else {
            // Calculate slice ranges
            let slice_ranges = match num_spatial_dims {
                1 => {
                    let p_begin = pads_begin[0];
                    let shape = image_shape[0];
                    let end = p_begin + shape;
                    quote! { [0..batch_size, 0..channels, #p_begin..#end] }
                }
                2 => {
                    let h_begin = pads_begin[0];
                    let h_shape = image_shape[0];
                    let h_end = h_begin + h_shape;

                    let w_begin = pads_begin[1];
                    let w_shape = image_shape[1];
                    let w_end = w_begin + w_shape;

                    quote! { [0..batch_size, 0..channels, #h_begin..#h_end, #w_begin..#w_end] }
                }
                _ => unreachable!("Unsupported dimensions checked by infer_types"),
            };

            quote! {
                 let canvas = canvas.reshape(#padded_shape_tokens);
                 canvas.slice(#slice_ranges)
            }
        };

        // Output image shape for result (validation/verification)
        // let output_shape = ...

        quote! {
            let #output = {
                let [batch_size, col_channels, _l] = #input.shape().dims();
                let channels = col_channels / #block_product;
                let device = #input.device();

                // 1. Convert input to flattened [N, C, BlockProd * Windows]
                // Note: col2im input is [N, C*Block, L], reshape to [N, C, Block*L] works if contiguous
                let input_flat = #input.reshape([batch_size, channels, #total_input_elements]);

                // 2. Create output canvas (Padded, Flattened)
                // Shape: [N, C, PaddedTotal]
                let canvas = Tensor::<B, 3>::zeros([batch_size, channels, #padded_size], &device);

                // 3. Create Indices Tensor [BlockProd * Windows]
                // We compute the indices at runtime using arange and broadcasting to avoid embedding large literal arrays.
                // Index = (window_pos * stride + block_pos * dilation) * stride_accumulator

                let indices = {
                    #indices_computation_code
                };

                // 4. Expand indices to [N, C, BlockProd * Windows]
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, #total_input_elements]);

                // 5. Scatter Add (dim 2)
                let canvas = canvas.scatter(2, indices_expanded, input_flat, burn::tensor::IndexingUpdateOp::Add);

                // 6. Reshape to Padded Spatial and Crop
                // If all pads are zero, we can skip the slice.
                #slice_logic
            };
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
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 4usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 64usize]);
                let canvas = Tensor::<B, 3>::zeros([batch_size, channels, 25usize], &device);
                let indices = {
                    let block_offsets = (Tensor::<B, 1, Int>::arange(0..2usize as i64, &device)
                        .mul_scalar(1i64)
                        .reshape([1usize, 2usize])
                        + Tensor::<B, 1, Int>::arange(0..2usize as i64, &device)
                            .mul_scalar(5i64)
                            .reshape([2usize, 1usize]))
                        .reshape([4usize, 1]);
                    let window_offsets = (Tensor::<B, 1, Int>::arange(0..4usize as i64, &device)
                        .mul_scalar(1i64)
                        .reshape([1usize, 4usize])
                        + Tensor::<B, 1, Int>::arange(0..4usize as i64, &device)
                            .mul_scalar(5i64)
                            .reshape([4usize, 1usize]))
                        .reshape([1, 16usize]);
                    (block_offsets + window_offsets).reshape([-1])
                };
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 64usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                canvas.reshape([batch_size, channels, 5usize, 5usize])
            };
            output
        }
        "###);
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
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 4usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 144usize]);
                let canvas = Tensor::<B, 3>::zeros([batch_size, channels, 49usize], &device);
                let indices = {
                    let block_offsets = (Tensor::<B, 1, Int>::arange(0..2usize as i64, &device)
                        .mul_scalar(1i64)
                        .reshape([1usize, 2usize])
                        + Tensor::<B, 1, Int>::arange(0..2usize as i64, &device)
                            .mul_scalar(7i64)
                            .reshape([2usize, 1usize]))
                        .reshape([4usize, 1]);
                    let window_offsets = (Tensor::<B, 1, Int>::arange(0..6usize as i64, &device)
                        .mul_scalar(1i64)
                        .reshape([1usize, 6usize])
                        + Tensor::<B, 1, Int>::arange(0..6usize as i64, &device)
                            .mul_scalar(7i64)
                            .reshape([6usize, 1usize]))
                        .reshape([1, 36usize]);
                    (block_offsets + window_offsets).reshape([-1])
                };
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 144usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                let canvas = canvas.reshape([batch_size, channels, 7usize, 7usize]);
                canvas.slice([0..batch_size, 0..channels, 1usize..6usize, 1usize..6usize])
            };
            output
        }
        "###);
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
        let node = Col2ImNodeBuilder::new("col2im_stride")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 4usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 36usize]);
                let canvas = Tensor::<B, 3>::zeros([batch_size, channels, 36usize], &device);
                let indices = {
                    let block_offsets = (Tensor::<B, 1, Int>::arange(0..2usize as i64, &device)
                        .mul_scalar(1i64)
                        .reshape([1usize, 2usize])
                        + Tensor::<B, 1, Int>::arange(0..2usize as i64, &device)
                            .mul_scalar(6i64)
                            .reshape([2usize, 1usize]))
                        .reshape([4usize, 1]);
                    let window_offsets = (Tensor::<B, 1, Int>::arange(0..3usize as i64, &device)
                        .mul_scalar(2i64)
                        .reshape([1usize, 3usize])
                        + Tensor::<B, 1, Int>::arange(0..3usize as i64, &device)
                            .mul_scalar(12i64)
                            .reshape([3usize, 1usize]))
                        .reshape([1, 9usize]);
                    (block_offsets + window_offsets).reshape([-1])
                };
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 36usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                canvas.reshape([batch_size, channels, 6usize, 6usize])
            };
            output
        }
        "###);
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
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 4usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 36usize]);
                let canvas = Tensor::<B, 3>::zeros([batch_size, channels, 25usize], &device);
                let indices = {
                    let block_offsets = (Tensor::<B, 1, Int>::arange(0..2usize as i64, &device)
                        .mul_scalar(2i64)
                        .reshape([1usize, 2usize])
                        + Tensor::<B, 1, Int>::arange(0..2usize as i64, &device)
                            .mul_scalar(10i64)
                            .reshape([2usize, 1usize]))
                        .reshape([4usize, 1]);
                    let window_offsets = (Tensor::<B, 1, Int>::arange(0..3usize as i64, &device)
                        .mul_scalar(1i64)
                        .reshape([1usize, 3usize])
                        + Tensor::<B, 1, Int>::arange(0..3usize as i64, &device)
                            .mul_scalar(5i64)
                            .reshape([3usize, 1usize]))
                        .reshape([1, 9usize]);
                    (block_offsets + window_offsets).reshape([-1])
                };
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 36usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                canvas.reshape([batch_size, channels, 5usize, 5usize])
            };
            output
        }
        "###);
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
        let node = Col2ImNodeBuilder::new("col2im1d")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 3usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 24usize]);
                let canvas = Tensor::<B, 3>::zeros([batch_size, channels, 10usize], &device);
                let indices = {
                    let block_offsets = (Tensor::<B, 1, Int>::arange(0..3usize as i64, &device)
                        .mul_scalar(1i64)
                        .reshape([3usize]))
                        .reshape([3usize, 1]);
                    let window_offsets = (Tensor::<B, 1, Int>::arange(0..8usize as i64, &device)
                        .mul_scalar(1i64)
                        .reshape([8usize]))
                        .reshape([1, 8usize]);
                    (block_offsets + window_offsets).reshape([-1])
                };
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 24usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                canvas.reshape([batch_size, channels, 10usize])
            };
            output
        }
        "###);
    }
}
