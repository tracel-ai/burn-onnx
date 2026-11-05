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

        // Burn's stft handles any n_fft: radix-2 FFT for powers of two, Bluestein otherwise.
        quote! {
            let #output = stft(
                { #signal_prep signal },
                #window_expr,
                burn::tensor::signal::StftOptions {
                    n_fft: #n_fft,
                    hop_length: #hop_length,
                    win_length: None,
                    center: false,
                    onesided: #onesided,
                },
            );
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::signal::stft");
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
    fn test_stft_non_pow2_onesided_no_window() {
        // Kokoro-style config: n_fft = 20 (not power of two), onesided.
        // Same burn stft call as the power-of-two case.
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
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<2>) -> Tensor<4> {
            let output = stft(
                {
                    let signal = signal;
                    signal
                },
                None,
                burn::tensor::signal::StftOptions {
                    n_fft: 20usize,
                    hop_length: 5usize,
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
            let output = stft(
                {
                    let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                    signal
                },
                None,
                burn::tensor::signal::StftOptions {
                    n_fft: {
                        let value = frame_length;
                        assert!(value > 0, "STFT frame_length must be positive");
                        usize::try_from(value).expect("STFT parameter does not fit usize")
                    },
                    hop_length: {
                        let value = frame_step;
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
