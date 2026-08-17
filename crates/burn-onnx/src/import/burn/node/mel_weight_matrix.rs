use super::prelude::*;

impl NodeCodegen for onnx_ir::node::mel_weight_matrix::MelWeightMatrixNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let num_mel_bins = scope.arg(&self.inputs[0]);
        let dft_length = scope.arg(&self.inputs[1]);
        let sample_rate = scope.arg(&self.inputs[2]);
        let lower_hz = scope.arg(&self.inputs[3]);
        let upper_hz = scope.arg(&self.inputs[4]);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let output_dtype = self.config.output_dtype.to_tokens();

        // Reproduces the ONNX reference algorithm, which differs from a "textbook" mel
        // filterbank: mel edges are snapped to integer DFT bin indices via
        // `floor((dft_length + 1) * hz / sample_rate)`, and the triangle ramps interpolate
        // between those integer indices (not between Hz values). This is why the official
        // test output has exact 1.0 peaks at DFT bins even when the true Hz peak falls
        // between bins.
        //
        // Note: `f32::log10`, `f32::powf`, and `f32::floor` are in `std::f32`. When the
        // surrounding model is generated with `LoadStrategy::File` (the default) the crate
        // root already emits `extern crate std;`, so these resolve. Generated models with
        // `LoadStrategy::None` in a `#![no_std]` consumer would fail to compile; that is a
        // broader limitation shared by any op that needs transcendental scalar math.
        quote! {
            let #output = {
                let num_mel_bins_i = #num_mel_bins as i64;
                let dft_length_i = (#dft_length) as i64;
                let sample_rate_i = (#sample_rate) as i64;
                let lower_edge_hertz_f = (#lower_hz) as f32;
                let upper_edge_hertz_f = (#upper_hz) as f32;

                // Surface clear errors for invalid runtime inputs instead of silently
                // producing a wrong matrix (negative dft_length collapses to shape [1, N]
                // of zeros), aborting the allocator (huge num_mel_bins), or diverging
                // inside the triangle loop (NaN from log10 of a non-positive argument).
                assert!(
                    num_mel_bins_i >= 0,
                    "MelWeightMatrix: num_mel_bins must be non-negative, got {}",
                    num_mel_bins_i,
                );
                assert!(
                    dft_length_i >= 0,
                    "MelWeightMatrix: dft_length must be non-negative, got {}",
                    dft_length_i,
                );
                assert!(
                    sample_rate_i > 0,
                    "MelWeightMatrix: sample_rate must be positive, got {}",
                    sample_rate_i,
                );
                assert!(
                    lower_edge_hertz_f >= 0.0_f32 && upper_edge_hertz_f >= 0.0_f32,
                    "MelWeightMatrix: edge_hertz values must be non-negative, \
                     got lower={} upper={}",
                    lower_edge_hertz_f,
                    upper_edge_hertz_f,
                );
                assert!(
                    lower_edge_hertz_f < upper_edge_hertz_f,
                    "MelWeightMatrix: lower_edge_hertz ({}) must be strictly less than \
                     upper_edge_hertz ({})",
                    lower_edge_hertz_f,
                    upper_edge_hertz_f,
                );

                let num_mel_bins = num_mel_bins_i as usize;
                let num_spectrogram_bins = (dft_length_i / 2 + 1) as usize;

                let low_mel = 2595.0_f32 * (1.0_f32 + lower_edge_hertz_f / 700.0_f32).log10();
                let high_mel = 2595.0_f32 * (1.0_f32 + upper_edge_hertz_f / 700.0_f32).log10();
                let n_edges = num_mel_bins + 2;
                // Note: ONNX reference divides by n_edges (N+2), not (N+1). Preserved here.
                let mel_step = (high_mel - low_mel) / (n_edges as f32);

                let bin_edges: alloc::vec::Vec<i64> = (0..n_edges)
                    .map(|i| {
                        let mel = low_mel + (i as f32) * mel_step;
                        let hz = 700.0_f32 * (10.0_f32.powf(mel / 2595.0_f32) - 1.0_f32);
                        ((dft_length_i + 1) as f32 * hz / sample_rate_i as f32).floor() as i64
                    })
                    .collect();

                let mut data: alloc::vec::Vec<f32> =
                    alloc::vec![0.0_f32; num_spectrogram_bins * num_mel_bins];
                for m in 0..num_mel_bins {
                    let lower = bin_edges[m];
                    let center = bin_edges[m + 1];
                    let upper = bin_edges[m + 2];
                    let low_to_center = center - lower;
                    if low_to_center == 0 {
                        // Degenerate rising ramp: set just the peak. The ONNX reference
                        // would index-error if `center` falls outside [0, num_spec_bins);
                        // we clamp instead because a user with an extreme Hz range on a
                        // small DFT can hit this path legitimately.
                        if center >= 0 && (center as usize) < num_spectrogram_bins {
                            data[center as usize * num_mel_bins + m] = 1.0_f32;
                        }
                    } else if low_to_center > 0 {
                        // Clamp in i64 space before casting to usize: a negative bin_edge
                        // would wrap to a huge value on a bare `as usize` cast, then
                        // .min(...) would leave hi pointing at the top of the matrix and
                        // fill every row with garbage weights.
                        let lo = lower.max(0) as usize;
                        let hi = center
                            .clamp(0, num_spectrogram_bins.saturating_sub(1) as i64)
                            as usize;
                        if hi >= lo && center >= 0 {
                            for j in lo..=hi {
                                data[j * num_mel_bins + m] =
                                    (j as f32 - lower as f32) / (low_to_center as f32);
                            }
                        }
                    }
                    let center_to_high = upper - center;
                    if center_to_high > 0 {
                        let lo = center.max(0) as usize;
                        let hi = upper.clamp(0, num_spectrogram_bins as i64) as usize;
                        if hi > lo && upper > 0 {
                            for j in lo..hi {
                                data[j * num_mel_bins + m] =
                                    (upper as f32 - j as f32) / (center_to_high as f32);
                            }
                        }
                    }
                }

                Tensor::<1>::from_floats(data.as_slice(), &self.device)
                    .reshape([num_spectrogram_bins, num_mel_bins])
                    .cast(#output_dtype)
            };
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // Only uses burn::prelude items and Tensor, already in scope via the generated module.
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::mel_weight_matrix::{MelWeightMatrixConfig, MelWeightMatrixNodeBuilder};

    #[test]
    fn test_mel_weight_matrix_runtime_f32() {
        let config = MelWeightMatrixConfig {
            output_dtype: DType::F32,
        };
        let node = MelWeightMatrixNodeBuilder::new("mwm1")
            .input_scalar("num_mel_bins", DType::I64)
            .input_scalar("dft_length", DType::I64)
            .input_scalar("sample_rate", DType::I64)
            .input_scalar("lower_edge_hertz", DType::F32)
            .input_scalar("upper_edge_hertz", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            num_mel_bins: i64,
            dft_length: i64,
            sample_rate: i64,
            lower_edge_hertz: f32,
            upper_edge_hertz: f32,
        ) -> Tensor<2> {
            let output = {
                let num_mel_bins_i = num_mel_bins as i64;
                let dft_length_i = (dft_length) as i64;
                let sample_rate_i = (sample_rate) as i64;
                let lower_edge_hertz_f = (lower_edge_hertz) as f32;
                let upper_edge_hertz_f = (upper_edge_hertz) as f32;
                assert!(
                    num_mel_bins_i >= 0,
                    "MelWeightMatrix: num_mel_bins must be non-negative, got {}", num_mel_bins_i,
                );
                assert!(
                    dft_length_i >= 0,
                    "MelWeightMatrix: dft_length must be non-negative, got {}", dft_length_i,
                );
                assert!(
                    sample_rate_i > 0, "MelWeightMatrix: sample_rate must be positive, got {}",
                    sample_rate_i,
                );
                assert!(
                    lower_edge_hertz_f >= 0.0_f32 && upper_edge_hertz_f >= 0.0_f32,
                    "MelWeightMatrix: edge_hertz values must be non-negative, \
                             got lower={} upper={}",
                    lower_edge_hertz_f, upper_edge_hertz_f,
                );
                assert!(
                    lower_edge_hertz_f < upper_edge_hertz_f,
                    "MelWeightMatrix: lower_edge_hertz ({}) must be strictly less than \
                             upper_edge_hertz ({})",
                    lower_edge_hertz_f, upper_edge_hertz_f,
                );
                let num_mel_bins = num_mel_bins_i as usize;
                let num_spectrogram_bins = (dft_length_i / 2 + 1) as usize;
                let low_mel = 2595.0_f32 * (1.0_f32 + lower_edge_hertz_f / 700.0_f32).log10();
                let high_mel = 2595.0_f32 * (1.0_f32 + upper_edge_hertz_f / 700.0_f32).log10();
                let n_edges = num_mel_bins + 2;
                let mel_step = (high_mel - low_mel) / (n_edges as f32);
                let bin_edges: alloc::vec::Vec<i64> = (0..n_edges)
                    .map(|i| {
                        let mel = low_mel + (i as f32) * mel_step;
                        let hz = 700.0_f32 * (10.0_f32.powf(mel / 2595.0_f32) - 1.0_f32);
                        ((dft_length_i + 1) as f32 * hz / sample_rate_i as f32).floor() as i64
                    })
                    .collect();
                let mut data: alloc::vec::Vec<f32> = alloc::vec![
                    0.0_f32; num_spectrogram_bins * num_mel_bins
                ];
                for m in 0..num_mel_bins {
                    let lower = bin_edges[m];
                    let center = bin_edges[m + 1];
                    let upper = bin_edges[m + 2];
                    let low_to_center = center - lower;
                    if low_to_center == 0 {
                        if center >= 0 && (center as usize) < num_spectrogram_bins {
                            data[center as usize * num_mel_bins + m] = 1.0_f32;
                        }
                    } else if low_to_center > 0 {
                        let lo = lower.max(0) as usize;
                        let hi = center.clamp(0, num_spectrogram_bins.saturating_sub(1) as i64)
                            as usize;
                        if hi >= lo && center >= 0 {
                            for j in lo..=hi {
                                data[j * num_mel_bins
                                    + m] = (j as f32 - lower as f32) / (low_to_center as f32);
                            }
                        }
                    }
                    let center_to_high = upper - center;
                    if center_to_high > 0 {
                        let lo = center.max(0) as usize;
                        let hi = upper.clamp(0, num_spectrogram_bins as i64) as usize;
                        if hi > lo && upper > 0 {
                            for j in lo..hi {
                                data[j * num_mel_bins
                                    + m] = (upper as f32 - j as f32) / (center_to_high as f32);
                            }
                        }
                    }
                }
                Tensor::<1>::from_floats(data.as_slice(), &self.device)
                    .reshape([num_spectrogram_bins, num_mel_bins])
                    .cast(burn::tensor::DType::F32)
            };
            output
        }
        "#);
    }

    #[test]
    fn test_mel_weight_matrix_runtime_f64() {
        let config = MelWeightMatrixConfig {
            output_dtype: DType::F64,
        };
        let node = MelWeightMatrixNodeBuilder::new("mwm1")
            .input_scalar("num_mel_bins", DType::I64)
            .input_scalar("dft_length", DType::I64)
            .input_scalar("sample_rate", DType::I64)
            .input_scalar("lower_edge_hertz", DType::F32)
            .input_scalar("upper_edge_hertz", DType::F32)
            .output_tensor("output", 2, DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        // Only the trailing .cast(...) differs from the F32 snapshot; asserting the full
        // expansion would be redundant, so pin just the dtype choice.
        assert!(code.contains("cast(burn::tensor::DType::F64)"));
    }
}
