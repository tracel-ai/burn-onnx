use super::prelude::*;
use onnx_ir::node::stft::StftConfig;

impl NodeCodegen for onnx_ir::node::stft::StftNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let signal_arg = &self.inputs[0];
        let output_arg = self.outputs.first().unwrap();

        let signal_tensor = match &signal_arg.ty {
            ArgType::Tensor(t) => t,
            other => unreachable!("STFT signal type validated in onnx-ir, got {other:?}"),
        };
        let signal_rank = signal_tensor.rank;

        let signal = scope.arg(signal_arg);
        let output = arg_to_ident(output_arg);

        let StftConfig {
            onesided,
            frame_step,
            frame_length,
            has_window,
        } = self.config;

        // Burn's `stft` takes a rank-2 real signal. Rank-3 ONNX inputs carry a
        // trailing 1 (real) or 2 (complex) dimension; onnx-ir rejects complex
        // so we only see real rank 3 here. Squeeze the trailing dim off; if
        // the signal is already rank 2 (PyTorch exporter shape), pass it through.
        let signal_prep = if signal_rank == 2 {
            quote! { let signal = #signal; }
        } else {
            let squeezed_rank = signal_rank - 1;
            let squeeze_dim = squeezed_rank as isize;
            quote! { let signal = #signal.squeeze_dims::<#squeezed_rank>(&[#squeeze_dim]); }
        };

        let window_tokens = if has_window {
            let window_arg = &self.inputs[2];
            let window = scope.arg(window_arg);
            Some(window)
        } else {
            None
        };

        let core = if frame_length.is_power_of_two() {
            pow2_core(frame_length, frame_step, onesided, window_tokens.as_ref())
        } else {
            matrix_dft_core(frame_length, frame_step, onesided, window_tokens.as_ref())
        };

        quote! {
            let #output = {
                #signal_prep
                #core
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // The fast path uses burn::tensor::signal::stft; the matmul path
        // doesn't need any special import beyond the usual Tensor/TensorData
        // that the prelude already brings in.
        imports.register("burn::tensor::signal::stft");
    }
}

/// Emit the fast path: upstream Burn's `stft` (requires power-of-two n_fft).
fn pow2_core(
    frame_length: usize,
    frame_step: usize,
    onesided: bool,
    window: Option<&TokenStream>,
) -> TokenStream {
    let window_expr = match window {
        Some(w) => quote! { Some(#w) },
        None => quote! { None },
    };
    quote! {
        let options = burn::tensor::signal::StftOptions {
            n_fft: #frame_length,
            hop_length: #frame_step,
            win_length: None,
            center: false,
            onesided: #onesided,
        };
        stft(signal, #window_expr, options)
    }
}

/// Emit a matrix DFT: works for any `n_fft` but O(N^2) per frame. Burn's
/// upstream `stft` is pow2-only (tracel-ai/burn#4865), so for non-pow2
/// models like Kokoro (n_fft=20) we compute the DFT as a matmul against
/// a precomputed twiddle-factor matrix.
fn matrix_dft_core(
    frame_length: usize,
    frame_step: usize,
    onesided: bool,
    window: Option<&TokenStream>,
) -> TokenStream {
    let n_freqs = if onesided {
        frame_length / 2 + 1
    } else {
        frame_length
    };

    // If no window, use a rank-1 ones tensor; otherwise take the caller's.
    // onnx-ir has already validated window shape == [frame_length].
    let window_bind = match window {
        Some(w) => quote! { let window: burn::tensor::Tensor<1> = #w; },
        None => quote! {
            let window: burn::tensor::Tensor<1> =
                burn::tensor::Tensor::ones([#frame_length], &device);
        },
    };

    quote! {
        let device = signal.device();

        // Frame the signal: [B, L] -> [B, n_frames, n_fft]
        let frames: burn::tensor::Tensor<3> =
            signal.unfold(1, #frame_length, #frame_step);

        // Apply window per frame.
        #window_bind
        let windowed: burn::tensor::Tensor<3> =
            frames.mul(window.reshape([1, 1, #frame_length]));

        // Cast to f64 for the matmul. The downstream graph (e.g. kokoro's
        // iSTFT preamble) often computes `imag/real` to recover phase,
        // which amplifies any f32 epsilon-level error in near-zero
        // spectral components into very large relative errors. Native FFT
        // backends like ORT's STFT happen to land near-zero values close
        // to the f32 zero; an O(N) f32 matmul does not, because partial
        // sums accumulate cancellation error of order O(N * eps) * max_val.
        // f64 matmul drops that error to f64 epsilon, which round-trips
        // through f32 cleanly for our scale.
        let windowed_f64 = windowed.cast(burn::tensor::DType::F64);

        // Compute DFT twiddle factors W[k, n] = exp(-j 2pi k n / N) in f64.
        // Conceptually constant, but kept at forward-call time on purpose:
        //   1. Generated code stays small regardless of n_fft (baking the
        //      twiddles as literals would add 2 * n_freqs * n_fft f64
        //      constants per STFT op, ballooning the .rs file and the
        //      snapshot tests).
        //   2. The matrix-DFT path only fires for non-pow2 n_fft, which is
        //      always small (large n_fft uses Burn's pow2 stft). For
        //      kokoro's n_fft=20 the cost is sub-microsecond per call.
        //   3. Once Bluestein's (tracel-ai/burn#4865) lands upstream, this
        //      whole path goes away.
        // If a future model makes this a real hot spot, the right fix is
        // upstream pow2-or-Bluestein support, not a per-codegen cache.
        let n_fft = #frame_length;
        let n_freqs = #n_freqs;
        let mut w_real: alloc::vec::Vec<f64> =
            alloc::vec::Vec::with_capacity(n_freqs * n_fft);
        let mut w_imag: alloc::vec::Vec<f64> =
            alloc::vec::Vec::with_capacity(n_freqs * n_fft);
        for k in 0..n_freqs {
            for n in 0..n_fft {
                let theta = 2.0_f64
                    * core::f64::consts::PI
                    * (k as f64)
                    * (n as f64)
                    / (n_fft as f64);
                w_real.push(theta.cos());
                w_imag.push(-theta.sin());
            }
        }
        // Pass (&device, DType::F64) so the tensor lands in f64; bare &device
        // would resolve to the backend's default float dtype (typically f32),
        // which would mismatch the f64 matmul below.
        let w_real_t: burn::tensor::Tensor<2> = burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(w_real, [n_freqs, n_fft]),
            (&device, burn::tensor::DType::F64),
        ).transpose();
        let w_imag_t: burn::tensor::Tensor<2> = burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(w_imag, [n_freqs, n_fft]),
            (&device, burn::tensor::DType::F64),
        ).transpose();

        // Flatten to [B*n_frames, n_fft], matmul in f64, then reshape back
        // and cast to f32 to match the ONNX STFT output dtype.
        let dims = windowed_f64.dims();
        let batch = dims[0];
        let n_frames = dims[1];
        let flat = windowed_f64.reshape([batch * n_frames, n_fft]);
        let re_f64 = flat.clone().matmul(w_real_t);
        let im_f64 = flat.matmul(w_imag_t);
        let re: burn::tensor::Tensor<3> =
            re_f64.reshape([batch, n_frames, n_freqs]).cast(burn::tensor::DType::F32);
        let im: burn::tensor::Tensor<3> =
            im_f64.reshape([batch, n_frames, n_freqs]).cast(burn::tensor::DType::F32);

        burn::tensor::Tensor::stack::<4>(alloc::vec![re, im], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::stft::{StftConfig, StftNodeBuilder};

    #[test]
    fn test_stft_onesided_no_window() {
        let config = StftConfig {
            onesided: true,
            frame_step: 4,
            frame_length: 16,
            has_window: false,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<3>) -> Tensor<4> {
            let output = {
                let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                let options = burn::tensor::signal::StftOptions {
                    n_fft: 16usize,
                    hop_length: 4usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                };
                stft(signal, None, options)
            };
            output
        }
        ");
    }

    #[test]
    fn test_stft_full_no_window() {
        let config = StftConfig {
            onesided: false,
            frame_step: 8,
            frame_length: 8,
            has_window: false,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<3>) -> Tensor<4> {
            let output = {
                let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                let options = burn::tensor::signal::StftOptions {
                    n_fft: 8usize,
                    hop_length: 8usize,
                    win_length: None,
                    center: false,
                    onesided: false,
                };
                stft(signal, None, options)
            };
            output
        }
        ");
    }

    #[test]
    fn test_stft_rank2_signal_no_squeeze() {
        // PyTorch's ONNX exporter emits a rank-2 signal (no trailing 1).
        // Burn's stft already wants rank 2, so codegen must not squeeze.
        let config = StftConfig {
            onesided: true,
            frame_step: 4,
            frame_length: 16,
            has_window: false,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 2, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<2>) -> Tensor<4> {
            let output = {
                let signal = signal;
                let options = burn::tensor::signal::StftOptions {
                    n_fft: 16usize,
                    hop_length: 4usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                };
                stft(signal, None, options)
            };
            output
        }
        ");
    }

    #[test]
    fn test_stft_non_pow2_onesided_no_window() {
        // Kokoro-style config: n_fft = 20 (not power of two), onesided.
        // Falls back to the matmul DFT path.
        let config = StftConfig {
            onesided: true,
            frame_step: 5,
            frame_length: 20,
            has_window: false,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 2, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<2>) -> Tensor<4> {
            let output = {
                let signal = signal;
                let device = signal.device();
                let frames: burn::tensor::Tensor<3> = signal.unfold(1, 20usize, 5usize);
                let window: burn::tensor::Tensor<1> = burn::tensor::Tensor::ones(
                    [20usize],
                    &device,
                );
                let windowed: burn::tensor::Tensor<3> = frames
                    .mul(window.reshape([1, 1, 20usize]));
                let windowed_f64 = windowed.cast(burn::tensor::DType::F64);
                let n_fft = 20usize;
                let n_freqs = 11usize;
                let mut w_real: alloc::vec::Vec<f64> = alloc::vec::Vec::with_capacity(
                    n_freqs * n_fft,
                );
                let mut w_imag: alloc::vec::Vec<f64> = alloc::vec::Vec::with_capacity(
                    n_freqs * n_fft,
                );
                for k in 0..n_freqs {
                    for n in 0..n_fft {
                        let theta = 2.0_f64 * core::f64::consts::PI * (k as f64) * (n as f64)
                            / (n_fft as f64);
                        w_real.push(theta.cos());
                        w_imag.push(-theta.sin());
                    }
                }
                let w_real_t: burn::tensor::Tensor<2> = burn::tensor::Tensor::from_data(
                        burn::tensor::TensorData::new(w_real, [n_freqs, n_fft]),
                        (&device, burn::tensor::DType::F64),
                    )
                    .transpose();
                let w_imag_t: burn::tensor::Tensor<2> = burn::tensor::Tensor::from_data(
                        burn::tensor::TensorData::new(w_imag, [n_freqs, n_fft]),
                        (&device, burn::tensor::DType::F64),
                    )
                    .transpose();
                let dims = windowed_f64.dims();
                let batch = dims[0];
                let n_frames = dims[1];
                let flat = windowed_f64.reshape([batch * n_frames, n_fft]);
                let re_f64 = flat.clone().matmul(w_real_t);
                let im_f64 = flat.matmul(w_imag_t);
                let re: burn::tensor::Tensor<3> = re_f64
                    .reshape([batch, n_frames, n_freqs])
                    .cast(burn::tensor::DType::F32);
                let im: burn::tensor::Tensor<3> = im_f64
                    .reshape([batch, n_frames, n_freqs])
                    .cast(burn::tensor::DType::F32);
                burn::tensor::Tensor::stack::<4>(alloc::vec![re, im], 3)
            };
            output
        }
        ");
    }

    #[test]
    fn test_stft_with_window() {
        let config = StftConfig {
            onesided: true,
            frame_step: 16,
            frame_length: 32,
            has_window: true,
        };
        // This test deliberately keeps frame_step as a dynamic input so it appears
        // in the forward() signature. In real models lift_constants folds it to
        // Static and it drops out; the builder helpers cannot produce Static args
        // directly, so we use a placeholder to exercise the window indexing logic.
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 3, DType::F32)
            .input_tensor("_frame_step_placeholder", 0, DType::I64)
            .input_tensor("window", 1, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            signal: Tensor<3>,
            _frame_step_placeholder: Tensor<0, Int>,
            window: Tensor<1>,
        ) -> Tensor<4> {
            let output = {
                let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                let options = burn::tensor::signal::StftOptions {
                    n_fft: 32usize,
                    hop_length: 16usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                };
                stft(signal, Some(window), options)
            };
            output
        }
        ");
    }
}
