use super::prelude::*;
use onnx_ir::node::stft::{StftConfig, StftInput};

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

        let StftConfig {
            onesided,
            frame_step,
            frame_length,
            has_window,
        } = &self.config;

        // Static values are usize literals; runtime ones are scalars read on the host.
        let mut usize_tokens = |input: &StftInput, parameter: &str| match input {
            StftInput::Static(v) => quote! { #v },
            StftInput::Runtime(r) => {
                let arg = &self.inputs[r.input_index];
                let value = match &arg.ty {
                    ArgType::ScalarTensor(dtype) => on_device_to_native(scope.arg(arg), dtype),
                    _ => scope.arg(arg),
                };
                let message = format!("STFT {parameter} must be positive");
                quote! {{
                    let value = #value;
                    assert!(value > 0, #message);
                    usize::try_from(value).expect("STFT parameter does not fit usize")
                }}
            }
        };
        let n_fft = usize_tokens(frame_length, "frame_length");
        let hop_length = usize_tokens(frame_step, "frame_step");

        let signal = scope.arg(signal_arg);
        let output = arg_to_ident(output_arg);

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

        let window_expr = if *has_window {
            let window = scope.arg(&self.inputs[2]);
            quote! { Some(#window) }
        } else {
            quote! { None }
        };

        // Keep radix-2 and large transforms on Burn's native FFT. Small
        // non-power-of-two windows are sensitive to cancellation in Bluestein's
        // f32 butterflies (notably Kokoro's 20-sample phase-recovery window).
        let native = quote! {
            stft(
                { #signal_prep signal },
                #window_expr,
                burn::tensor::signal::StftOptions {
                    n_fft: #n_fft,
                    hop_length: #hop_length,
                    win_length: None,
                    center: false,
                    onesided: #onesided,
                },
            )
        };
        if matches!(frame_length, StftInput::Static(n) if n.is_power_of_two() || *n > 32) {
            return quote! { let #output = #native; };
        }

        let dtype = signal_tensor.dtype.to_tokens();
        let dft = small_dft_core(*onesided, dtype);
        let core = if matches!(frame_length, StftInput::Runtime(_)) {
            quote! {
                if n_fft <= 32 && !n_fft.is_power_of_two() {
                    #dft
                } else {
                    stft(signal, window, burn::tensor::signal::StftOptions {
                        n_fft, hop_length, win_length: None, center: false, onesided: #onesided,
                    })
                }
            }
        } else {
            dft
        };
        quote! {
            let #output = {
                let (signal, window, n_fft, hop_length): (_, Option<Tensor<1>>, usize, usize) =
                    ({ #signal_prep signal }, #window_expr, #n_fft, #hop_length);
                #core
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if !matches!(&self.config.frame_length, StftInput::Static(n) if !n.is_power_of_two() && *n <= 32)
        {
            imports.register("burn::tensor::signal::stft");
        }
        if !matches!(&self.config.frame_length, StftInput::Static(n) if n.is_power_of_two() || *n > 32)
        {
            imports.register("burn::tensor::DType");
        }
    }
}

/// A bounded direct DFT: at most 32 x 32 coefficients, accumulated in f64 when
/// available. The frame/window product retains the ONNX input dtype, and the
/// result is cast back to it. Large windows keep the O(N log N) native FFT.
fn small_dft_core(onesided: bool, output_dtype: TokenStream) -> TokenStream {
    quote! {
        // Preserve the frame/hop contract enforced by the IR for constants and
        // by Burn's native STFT for runtime parameters on the FFT path.
        assert!(hop_length <= n_fft, "STFT frame_step must not exceed frame_length");
        let device = signal.device();
        let frames: Tensor<3> = signal.unfold(1, n_fft, hop_length);
        let windowed = if let Some(window) = window {
            frames.mul(window.reshape([1, 1, n_fft]))
        } else {
            frames
        };
        let dft_dtype = if device.supports_dtype(DType::F64) { DType::F64 } else { DType::F32 };
        let windowed = windowed.cast(dft_dtype);
        let n_freqs = if #onesided { n_fft / 2 + 1 } else { n_fft };
        let mut angles = alloc::vec::Vec::with_capacity(n_freqs * n_fft);
        for k in 0..n_freqs {
            for n in 0..n_fft {
                angles.push(2.0_f64 * core::f64::consts::PI * k as f64 * n as f64 / n_fft as f64);
            }
        }
        let angles: Tensor<2> = Tensor::from_data(
            TensorData::new(angles, [n_freqs, n_fft]), (&device, dft_dtype),
        ).transpose();
        let real = angles.clone().cos();
        let imag = angles.sin().neg();
        let [batch, n_frames, _] = windowed.dims();
        let flat = windowed.reshape([batch * n_frames, n_fft]);
        let real: Tensor<3> = flat.clone().matmul(real)
            .reshape([batch, n_frames, n_freqs]).cast(#output_dtype);
        let imag: Tensor<3> = flat.matmul(imag)
            .reshape([batch, n_frames, n_freqs]).cast(#output_dtype);
        Tensor::stack::<4>(alloc::vec![real, imag], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::node::stft::{StftConfig, StftInput, StftNodeBuilder};

    #[test]
    fn test_stft_onesided_no_window() {
        let config = StftConfig {
            onesided: true,
            frame_step: StftInput::Static(4),
            frame_length: StftInput::Static(16),
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
            let output = stft(
                {
                    let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                    signal
                },
                None,
                burn::tensor::signal::StftOptions {
                    n_fft: 16usize,
                    hop_length: 4usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                },
            );
            output
        }
        ");
    }

    #[test]
    fn test_stft_full_no_window() {
        let config = StftConfig {
            onesided: false,
            frame_step: StftInput::Static(8),
            frame_length: StftInput::Static(8),
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
            let output = stft(
                {
                    let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                    signal
                },
                None,
                burn::tensor::signal::StftOptions {
                    n_fft: 8usize,
                    hop_length: 8usize,
                    win_length: None,
                    center: false,
                    onesided: false,
                },
            );
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
            frame_step: StftInput::Static(4),
            frame_length: StftInput::Static(16),
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
            let output = stft(
                {
                    let signal = signal;
                    signal
                },
                None,
                burn::tensor::signal::StftOptions {
                    n_fft: 16usize,
                    hop_length: 4usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                },
            );
            output
        }
        ");
    }

    #[test]
    fn test_stft_small_full_f64_with_window() {
        let node = StftNodeBuilder::new("stft")
            .input_tensor("signal", 2, DType::F64)
            .input_scalar("step", DType::I64)
            .input_tensor("window", 1, DType::F64)
            .output_tensor("output", 4, DType::F64)
            .config(StftConfig {
                onesided: false,
                frame_step: StftInput::Static(5),
                frame_length: StftInput::Static(20),
                has_window: true,
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, signal: Tensor<2>, step: i64, window: Tensor<1>) -> Tensor<4> {
            let output = {
                let (signal, window, n_fft, hop_length): (_, Option<Tensor<1>>, usize, usize) = (
                    {
                        let signal = signal;
                        signal
                    },
                    Some(window),
                    20usize,
                    5usize,
                );
                assert!(hop_length <= n_fft, "STFT frame_step must not exceed frame_length");
                let device = signal.device();
                let frames: Tensor<3> = signal.unfold(1, n_fft, hop_length);
                let windowed = if let Some(window) = window {
                    frames.mul(window.reshape([1, 1, n_fft]))
                } else {
                    frames
                };
                let dft_dtype = if device.supports_dtype(DType::F64) {
                    DType::F64
                } else {
                    DType::F32
                };
                let windowed = windowed.cast(dft_dtype);
                let n_freqs = if false { n_fft / 2 + 1 } else { n_fft };
                let mut angles = alloc::vec::Vec::with_capacity(n_freqs * n_fft);
                for k in 0..n_freqs {
                    for n in 0..n_fft {
                        angles
                            .push(
                                2.0_f64 * core::f64::consts::PI * k as f64 * n as f64
                                    / n_fft as f64,
                            );
                    }
                }
                let angles: Tensor<2> = Tensor::from_data(
                        TensorData::new(angles, [n_freqs, n_fft]),
                        (&device, dft_dtype),
                    )
                    .transpose();
                let real = angles.clone().cos();
                let imag = angles.sin().neg();
                let [batch, n_frames, _] = windowed.dims();
                let flat = windowed.reshape([batch * n_frames, n_fft]);
                let real: Tensor<3> = flat
                    .clone()
                    .matmul(real)
                    .reshape([batch, n_frames, n_freqs])
                    .cast(burn::tensor::DType::F64);
                let imag: Tensor<3> = flat
                    .matmul(imag)
                    .reshape([batch, n_frames, n_freqs])
                    .cast(burn::tensor::DType::F64);
                Tensor::stack::<4>(alloc::vec![real, imag], 3)
            };
            output
        }
        "#);
    }

    #[test]
    fn test_stft_large_non_pow2_uses_native_fft() {
        let node = StftNodeBuilder::new("stft")
            .input_tensor("signal", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(StftConfig {
                onesided: true,
                frame_step: StftInput::Static(400),
                frame_length: StftInput::Static(400),
                has_window: false,
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<3>) -> Tensor<4> {
            let output = stft(
                {
                    let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                    signal
                },
                None,
                burn::tensor::signal::StftOptions {
                    n_fft: 400usize,
                    hop_length: 400usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                },
            );
            output
        }
        ");
    }

    #[test]
    fn test_stft_non_pow2_onesided_no_window() {
        // Kokoro-style config: n_fft = 20 (not power of two), onesided.
        // Small windows use the bounded high-precision DFT.
        let config = StftConfig {
            onesided: true,
            frame_step: StftInput::Static(5),
            frame_length: StftInput::Static(20),
            has_window: false,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 2, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, signal: Tensor<2>) -> Tensor<4> {
            let output = {
                let (signal, window, n_fft, hop_length): (_, Option<Tensor<1>>, usize, usize) = (
                    {
                        let signal = signal;
                        signal
                    },
                    None,
                    20usize,
                    5usize,
                );
                assert!(hop_length <= n_fft, "STFT frame_step must not exceed frame_length");
                let device = signal.device();
                let frames: Tensor<3> = signal.unfold(1, n_fft, hop_length);
                let windowed = if let Some(window) = window {
                    frames.mul(window.reshape([1, 1, n_fft]))
                } else {
                    frames
                };
                let dft_dtype = if device.supports_dtype(DType::F64) {
                    DType::F64
                } else {
                    DType::F32
                };
                let windowed = windowed.cast(dft_dtype);
                let n_freqs = if true { n_fft / 2 + 1 } else { n_fft };
                let mut angles = alloc::vec::Vec::with_capacity(n_freqs * n_fft);
                for k in 0..n_freqs {
                    for n in 0..n_fft {
                        angles
                            .push(
                                2.0_f64 * core::f64::consts::PI * k as f64 * n as f64
                                    / n_fft as f64,
                            );
                    }
                }
                let angles: Tensor<2> = Tensor::from_data(
                        TensorData::new(angles, [n_freqs, n_fft]),
                        (&device, dft_dtype),
                    )
                    .transpose();
                let real = angles.clone().cos();
                let imag = angles.sin().neg();
                let [batch, n_frames, _] = windowed.dims();
                let flat = windowed.reshape([batch * n_frames, n_fft]);
                let real: Tensor<3> = flat
                    .clone()
                    .matmul(real)
                    .reshape([batch, n_frames, n_freqs])
                    .cast(burn::tensor::DType::F32);
                let imag: Tensor<3> = flat
                    .matmul(imag)
                    .reshape([batch, n_frames, n_freqs])
                    .cast(burn::tensor::DType::F32);
                Tensor::stack::<4>(alloc::vec![real, imag], 3)
            };
            output
        }
        "#);
    }

    #[test]
    fn test_stft_with_window() {
        let config = StftConfig {
            onesided: true,
            frame_step: StftInput::Static(16),
            frame_length: StftInput::Static(32),
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
            let output = stft(
                {
                    let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                    signal
                },
                Some(window),
                burn::tensor::signal::StftOptions {
                    n_fft: 32usize,
                    hop_length: 16usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                },
            );
            output
        }
        ");
    }

    #[test]
    fn test_stft_runtime_frame_step_and_length() {
        // ONNX backend test `test_stft`: frame_step and frame_length are graph inputs.
        let config = StftConfig {
            onesided: true,
            frame_step: StftInput::Runtime(RuntimeInputRef::new("frame_step".to_string(), 1)),
            frame_length: StftInput::Runtime(RuntimeInputRef::new("frame_length".to_string(), 2)),
            has_window: false,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 3, DType::F32)
            .input_scalar("frame_step", DType::I64)
            .input_scalar("frame_length", DType::I32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            signal: Tensor<3>,
            frame_step: i64,
            frame_length: i32,
        ) -> Tensor<4> {
            let output = {
                let (signal, window, n_fft, hop_length): (_, Option<Tensor<1>>, usize, usize) = (
                    {
                        let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                        signal
                    },
                    None,
                    {
                        let value = frame_length;
                        assert!(value > 0, "STFT frame_length must be positive");
                        usize::try_from(value).expect("STFT parameter does not fit usize")
                    },
                    {
                        let value = frame_step;
                        assert!(value > 0, "STFT frame_step must be positive");
                        usize::try_from(value).expect("STFT parameter does not fit usize")
                    },
                );
                if n_fft <= 32 && !n_fft.is_power_of_two() {
                    assert!(hop_length <= n_fft, "STFT frame_step must not exceed frame_length");
                    let device = signal.device();
                    let frames: Tensor<3> = signal.unfold(1, n_fft, hop_length);
                    let windowed = if let Some(window) = window {
                        frames.mul(window.reshape([1, 1, n_fft]))
                    } else {
                        frames
                    };
                    let dft_dtype = if device.supports_dtype(DType::F64) {
                        DType::F64
                    } else {
                        DType::F32
                    };
                    let windowed = windowed.cast(dft_dtype);
                    let n_freqs = if true { n_fft / 2 + 1 } else { n_fft };
                    let mut angles = alloc::vec::Vec::with_capacity(n_freqs * n_fft);
                    for k in 0..n_freqs {
                        for n in 0..n_fft {
                            angles
                                .push(
                                    2.0_f64 * core::f64::consts::PI * k as f64 * n as f64
                                        / n_fft as f64,
                                );
                        }
                    }
                    let angles: Tensor<2> = Tensor::from_data(
                            TensorData::new(angles, [n_freqs, n_fft]),
                            (&device, dft_dtype),
                        )
                        .transpose();
                    let real = angles.clone().cos();
                    let imag = angles.sin().neg();
                    let [batch, n_frames, _] = windowed.dims();
                    let flat = windowed.reshape([batch * n_frames, n_fft]);
                    let real: Tensor<3> = flat
                        .clone()
                        .matmul(real)
                        .reshape([batch, n_frames, n_freqs])
                        .cast(burn::tensor::DType::F32);
                    let imag: Tensor<3> = flat
                        .matmul(imag)
                        .reshape([batch, n_frames, n_freqs])
                        .cast(burn::tensor::DType::F32);
                    Tensor::stack::<4>(alloc::vec![real, imag], 3)
                } else {
                    stft(
                        signal,
                        window,
                        burn::tensor::signal::StftOptions {
                            n_fft,
                            hop_length,
                            win_length: None,
                            center: false,
                            onesided: true,
                        },
                    )
                }
            };
            output
        }
        "#);
    }

    #[test]
    fn test_stft_runtime_frame_step_scalar_tensor_with_window() {
        // An on-device frame_step is read back to the host.
        let config = StftConfig {
            onesided: true,
            frame_step: StftInput::Runtime(RuntimeInputRef::new("frame_step".to_string(), 1)),
            frame_length: StftInput::Static(16),
            has_window: true,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 3, DType::F32)
            .input_scalar_tensor("frame_step", DType::I64)
            .input_tensor("window", 1, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            signal: Tensor<3>,
            frame_step: Tensor<1, Int>,
            window: Tensor<1>,
        ) -> Tensor<4> {
            let output = stft(
                {
                    let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                    signal
                },
                Some(window),
                burn::tensor::signal::StftOptions {
                    n_fft: 16usize,
                    hop_length: {
                        let value = (frame_step).into_scalar::<i64>();
                        assert!(value > 0, "STFT frame_step must be positive");
                        usize::try_from(value).expect("STFT parameter does not fit usize")
                    },
                    win_length: None,
                    center: false,
                    onesided: true,
                },
            );
            output
        }
        "#);
    }
}
