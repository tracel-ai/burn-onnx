use super::prelude::*;
use onnx_ir::node::dft::DftConfig;

impl NodeCodegen for onnx_ir::node::dft::DftNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        let input_tensor = match &input_arg.ty {
            ArgType::Tensor(t) => t,
            other => unreachable!("DFT input type validated in onnx-ir, got {other:?}"),
        };

        let input_rank = input_tensor.rank;
        let config = &self.config;

        // Unsupported configurations (inverse, complex input) are rejected
        // in onnx-ir's infer_types with ProcessError. These branches are unreachable.
        if config.onesided {
            forward_rfft(config, input, output, input_rank)
        } else {
            forward_rfft_full(config, input, output, input_rank)
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::signal::rfft");
    }
}

/// Forward real DFT with onesided output via rfft
///
/// ONNX: [..., N, 1] -> [..., N/2+1, 2]
fn forward_rfft(
    config: &DftConfig,
    input: TokenStream,
    output: Ident,
    input_rank: usize,
) -> TokenStream {
    // After squeezing trailing [1] dim: signal has rank = input_rank - 1
    let signal_rank = input_rank - 1;
    let out_rank = input_rank; // output [..., K, 2] same rank as input [..., N, 1]
    let axis = config.axis;
    // squeeze_dims takes &[isize]
    let squeeze_dim = signal_rank as isize;

    let dft_length_code = dft_length_adjustment(config, axis);

    quote! {
        let #output = {
            let signal = #input.squeeze_dims::<#signal_rank>(&[#squeeze_dim]);
            #dft_length_code
            let (re, im) = rfft(signal, #axis);
            // Stack re and im along new last dim: [.., K] + [.., K] -> [.., K, 2]
            Tensor::<B, #signal_rank>::stack::<#out_rank>(
                [re, im].to_vec(),
                #signal_rank,
            )
        };
    }
}

/// Forward real DFT with full (non-onesided) output
///
/// ONNX: [..., N, 1] -> [..., N, 2]
///
/// Computes rfft (onesided), then reconstructs the full spectrum
/// using conjugate symmetry: X[N-k] = conj(X[k])
fn forward_rfft_full(
    config: &DftConfig,
    input: TokenStream,
    output: Ident,
    input_rank: usize,
) -> TokenStream {
    let signal_rank = input_rank - 1;
    let out_rank = input_rank;
    let axis = config.axis;
    let squeeze_dim = signal_rank as isize;
    let flip_axis = axis as isize; // flip() takes &[isize]

    let dft_length_code = dft_length_adjustment(config, axis);

    quote! {
        let #output = {
            let signal = #input.squeeze_dims::<#signal_rank>(&[#squeeze_dim]);
            #dft_length_code
            let n = signal.dims()[#axis];
            let (re_half, im_half) = rfft(signal, #axis);
            let half_len = re_half.dims()[#axis];
            let mirror_len = n - half_len;

            if mirror_len == 0 {
                // Degenerate case (n <= 1): no mirrored bins to reconstruct
                Tensor::<B, #signal_rank>::stack::<#out_rank>(
                    [re_half, im_half].to_vec(),
                    #signal_rank,
                )
            } else {
                // Mirror conjugate symmetry: X[N-k] = conj(X[k])
                let mirror_re = re_half
                    .clone()
                    .narrow(#axis, 1, mirror_len)
                    .flip([#flip_axis]);
                let mirror_im = im_half
                    .clone()
                    .narrow(#axis, 1, mirror_len)
                    .flip([#flip_axis])
                    .neg();

                let re_full = Tensor::<B, #signal_rank>::cat(
                    [re_half, mirror_re].to_vec(), #axis,
                );
                let im_full = Tensor::<B, #signal_rank>::cat(
                    [im_half, mirror_im].to_vec(), #axis,
                );

                Tensor::<B, #signal_rank>::stack::<#out_rank>(
                    [re_full, im_full].to_vec(),
                    #signal_rank,
                )
            }
        };
    }
}

/// Generate code for dft_length adjustment (zero-padding or truncation)
fn dft_length_adjustment(config: &DftConfig, axis: usize) -> TokenStream {
    match config.dft_length {
        Some(dft_length) => {
            quote! {
                let signal = {
                    let current_len = signal.dims()[#axis];
                    if current_len < #dft_length {
                        let mut pad_shape = signal.dims();
                        pad_shape[#axis] = #dft_length - current_len;
                        let padding = Tensor::zeros(pad_shape, &signal.device());
                        Tensor::cat([signal, padding].to_vec(), #axis)
                    } else if current_len > #dft_length {
                        signal.narrow(#axis, 0, #dft_length)
                    } else {
                        signal
                    }
                };
            }
        }
        None => quote! {},
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::dft::{DftConfig, DftNodeBuilder};

    #[test]
    fn test_dft_forward_onesided_real() {
        let config = DftConfig {
            inverse: false,
            onesided: true,
            axis: 1,
            dft_length: None,
            is_real_input: true,
        };
        let node = DftNodeBuilder::new("dft1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let signal = input.squeeze_dims::<2usize>(&[2isize]);
                let (re, im) = rfft(signal, 1usize);
                Tensor::<B, 2usize>::stack::<3usize>([re, im].to_vec(), 2usize)
            };
            output
        }
        ");
    }

    #[test]
    fn test_dft_forward_full_real() {
        let config = DftConfig {
            inverse: false,
            onesided: false,
            axis: 1,
            dft_length: None,
            is_real_input: true,
        };
        let node = DftNodeBuilder::new("dft1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let signal = input.squeeze_dims::<2usize>(&[2isize]);
                let n = signal.dims()[1usize];
                let (re_half, im_half) = rfft(signal, 1usize);
                let half_len = re_half.dims()[1usize];
                let mirror_len = n - half_len;
                if mirror_len == 0 {
                    Tensor::<B, 2usize>::stack::<3usize>([re_half, im_half].to_vec(), 2usize)
                } else {
                    let mirror_re = re_half.clone().narrow(1usize, 1, mirror_len).flip([1isize]);
                    let mirror_im = im_half
                        .clone()
                        .narrow(1usize, 1, mirror_len)
                        .flip([1isize])
                        .neg();
                    let re_full = Tensor::<
                        B,
                        2usize,
                    >::cat([re_half, mirror_re].to_vec(), 1usize);
                    let im_full = Tensor::<
                        B,
                        2usize,
                    >::cat([im_half, mirror_im].to_vec(), 1usize);
                    Tensor::<B, 2usize>::stack::<3usize>([re_full, im_full].to_vec(), 2usize)
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_dft_forward_onesided_with_dft_length() {
        let config = DftConfig {
            inverse: false,
            onesided: true,
            axis: 1,
            dft_length: Some(32),
            is_real_input: true,
        };
        let node = DftNodeBuilder::new("dft1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let signal = input.squeeze_dims::<2usize>(&[2isize]);
                let signal = {
                    let current_len = signal.dims()[1usize];
                    if current_len < 32usize {
                        let mut pad_shape = signal.dims();
                        pad_shape[1usize] = 32usize - current_len;
                        let padding = Tensor::zeros(pad_shape, &signal.device());
                        Tensor::cat([signal, padding].to_vec(), 1usize)
                    } else if current_len > 32usize {
                        signal.narrow(1usize, 0, 32usize)
                    } else {
                        signal
                    }
                };
                let (re, im) = rfft(signal, 1usize);
                Tensor::<B, 2usize>::stack::<3usize>([re, im].to_vec(), 2usize)
            };
            output
        }
        ");
    }
}
