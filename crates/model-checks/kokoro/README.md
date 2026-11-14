# Kokoro TTS Model Check

This model check exercises burn-onnx against [Kokoro v1.0][src], an 82M-parameter
text-to-speech model that produces 24 kHz audio from phoneme IDs and a 256-d
voice-style embedding.

[src]: https://github.com/thewh1teagle/kokoro-onnx

## Model

- **File**: `kokoro-v1.0.onnx` (~310 MB, opset 20, PyTorch 2.6.0 export)
- **Inputs**:
  - `tokens`: `int64[1, sequence_length]` phoneme IDs (vocab 178, id 0 is pad/boundary)
  - `style`: `float32[1, 256]` voice style embedding
  - `speed`: `float32[1]` playback rate (1.0 = nominal)
- **Output**: `float32[audio_length]` raw waveform at 24 kHz
- **Graph**: 2,464 nodes, 49 distinct op types including `STFT`, `LSTM`,
  `ConvTranspose`, `ScatterND`, `NonZero`, `Resize`, `CumSum`.

## Current Status

**Runs end-to-end, but still fails strict numerical comparison (#371).**
The normal check loads `kokoro-v1.0.rs` and `kokoro-v1.0.bpk`, matching the
complete ONNX file stem. The default smoke threshold is not a numerical match.

### Small-window precision regression (2026-11-09)

On main `77c18b6d` / Burn `02d45852`, native f32 FFT for the 20-sample
STFT window fails even the smoke check. A bounded direct DFT restores the
previous numerical behavior without sending large windows through a quadratic
algorithm. Non-power-of-two windows of at most 32 samples use f64 accumulation
when the device supports it (f32 otherwise); larger windows and powers of two
keep the native FFT. Runtime frame lengths use the same selection.

Measured with Flex on Apple Silicon in release mode, using the original model
and a freshly generated seed-42 ORT reference:

| Implementation | Burn peak | Pearson r | Max absolute error | Mean absolute error |
|---|---:|---:|---:|---:|
| Native f32 FFT (main) | 1113.1952 | 0.405139 | 1096.1067 | 5.217939 |
| Bounded f64 matrix DFT | 224.9446 | 0.748335 | 155.3031 | 1.733947 |

The second row passes the unchanged smoke threshold. **It still fails strict
comparison**, whose max/mean limits remain 1.3078/0.1308. This repairs the smoke
regression but does not resolve the residual phase sensitivity in #371. The
manual phase graph is unchanged; replacing it with `atan2` is not equivalent
at signed zero. No tolerances were relaxed.

Reproduce with:

```sh
uv run get_model.py
cargo run --release --bin burn-onnx-model-checks-kokoro
KOKORO_STRICT=1 cargo run --release --bin burn-onnx-model-checks-kokoro
```

### FFT experiments (2026-11-05)

Measured on Apple Silicon/macOS with the Flex backend, a release build, main
`9c04afa5`, Burn `26d3257e`, ONNX 1.19.0, and ONNX Runtime 1.19.2 CPU. Inputs
are the existing seed-42, 10-token case from `get_model.py` (13,200 samples).
ORT's original-model peak is 130.7800. The model SHA-256 is
`7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5`.

| STFT / phase calculation | Burn peak | Pearson r | Max absolute error | Mean absolute error |
|---|---:|---:|---:|---:|
| Main: f64 matrix DFT / original phase | 224.9446 | 0.748335 | 155.3031 | 1.733947 |
| Native f32 FFT / original phase | 1113.1952 | 0.405139 | 1096.1067 | 5.217939 |
| Native f64 FFT / original phase | 655.5057 | 0.825044 | 541.4872 | 2.311333 |
| Native f32 FFT / `atan2` | 428.4377 | 0.655831 | 411.3493 | 3.119516 |
| Native f64 FFT / `atan2` | 655.5063 | 0.825044 | 541.4872 | 2.311338 |

All five variants fail the unchanged strict limits (max 1.3078, mean 0.1308).
Native f32 FFT also fails the smoke correlation floor of 0.5. Therefore simply
removing the matrix DFT, as proposed in [PR #674](https://github.com/tracel-ai/burn-onnx/pull/674),
does **not** resolve this model's divergence. These experimental code changes
are not part of this check.

To reproduce the native experiment, apply the STFT IR and codegen files from
PR #674 (`fb12a137`) and run the normal check. The f64 experiment casts both the
signal and window to F64 before `stft`, then casts its result back to F32. The
phase experiment replaces the final manual phase result with
`imag.atan2(real)`; it is a diagnostic substitution, not a semantics-preserving
ONNX rewrite. In particular, for negative real and positive zero imaginary,
the graph's `imag > 0` branch returns -pi while `atan2` returns +pi.

Instrumentation itself also changes ORT results: adding the bisection taps
changes its peak to 150.1652 for the same input. `bisect_setup.py` now runs
both original and instrumented graphs and records their difference in
`bisect_taps.json`. Tap statistics describe the instrumented graph; equal
summary statistics do not imply element-wise equality. A successful fix still
needs strict validation against the original model, without relaxing tolerances
or altering phase semantics.

### Debug history

The `bisect_setup.py` + `src/bin/bisect.rs` (gated behind the `bisect`
feature) infrastructure modifies the kokoro ONNX to expose ~190
intermediate tensors and compares Burn's outputs to ORT layer-by-layer.

The first round of bisection found matching upstream summary statistics
through `ups.0`; the smoking gun was `/decoder/decoder/generator/Div`
(idx 1582), which computes `imag/real` of the STFT output to recover phase.
Both Burn and ORT agreed on STFT summary stats (`abs_max`, `abs_mean`
identical to 4 decimal places), but the matrix-DFT we emit for non-power-of-two
`n_fft` accumulates O(N) f32 summation error, which makes near-zero
spectral components diverge from ORT's by f32 epsilon. Dividing by those
near-zero values amplified the f32-epsilon error into 24% relative error,
which then propagated through `Exp` (log-magnitude → linear magnitude) as
a 20-25× spike in the resulting audio.

**Fix:** the matrix-DFT path in `crates/burn-onnx/src/import/burn/node/stft.rs`
now casts the windowed signal to f64, computes twiddles in f64, performs
the matmul in f64, and casts back to f32. f64 epsilon is small enough that
near-zero spectrum elements stay close to zero, so the downstream
`imag/real` does not amplify into a 20× linear blowup.

| Metric | Before f64 fix | After f64 fix |
|---|---|---|
| Burn audio `abs_max` | 3,109 | ~170 |
| ORT audio `abs_max` | ~131 | ~131 |
| Pearson r (Burn vs ORT) | 0.39 | 0.69 |
| Best-fit linear scale (Burn = scale * ORT) | 9.06× | 1.51× |
| `max |Burn − ORT|` | 3,087 | ~156 |

### Remaining divergence

The remaining 1.3× peak divergence (Pearson r 0.69) is **not** generic f32
drift through the resblock stack — that earlier hypothesis was wrong, the
abs_max metric was just shifting around. The real cause is a manual
`atan2` decomposition in the kokoro graph (`Atan + Div + Greater + Less +
Where`) interacting badly with sign-of-zero in the STFT output:

- Element-wise comparison shows Burn's STFT `imag` has **2,641 exact zeros**
  (DC bin, where `sin(0)·signal = 0` analytically) while ORT has **3 exact
  zeros**. ORT's f32 FFT propagates rounding noise to true-zero positions;
  Burn's matrix-DFT (any precision) computes them as exact zero.
- For those positions, Burn's `Greater(imag, 0)` returns `false` while ORT's
  returns true/false depending on its noise sign, picking different `Where`
  branches and shifting phase by ±π. **9,424 of 29,051 Atan output elements
  (~32%) end up differing by more than 0.01 rad, with `max |Δ| = π`.**
- Those phase differences feed `noise_convs.0` and propagate through the
  generator stack into the audio, surfacing as the residual 1.3× peak
  factor.

This is a fundamental algorithm divergence, not a Burn bug — see
[burn-onnx#371][issue-371] for the full investigation, element-wise data,
and possible fixes (manual-atan2 → atan2 coalescer pass, or implementing
Bluestein's algorithm upstream).

### Atan2 coalescing decision

The manual `Atan + Div + Greater + Less + Where` sequence was evaluated as a
candidate simplification. It is not safe to replace this graph with native
`atan2` without a proof that the real and imaginary inputs cannot reach signed
zero: for a negative real input, the manual quadrant branch and native
`atan2(+0, negative)` produce opposite signed phase values (`-pi` versus
`+pi`). The Kokoro graph has dynamic STFT-derived inputs, so onnx-ir cannot
establish that proof from its current type and shape information.

The candidate was also measured in the existing Kokoro experiments above; the
native-`atan2` substitutions remained far outside the strict comparison limits.
This change therefore deliberately does not add an unsafe coalescing rewrite.

[issue-371]: https://github.com/tracel-ai/burn-onnx/issues/371

### Previously fixed while building this check (still relevant)

- **onnx-ir rejected rank-2 STFT signal input.** PyTorch's ONNX exporter
  emits the signal as rank 2 `[batch, signal_length]`. The onnx-ir STFT
  processor required the spec's rank 3 `[batch, signal_length, 1|2]`, which
  blocked every PyTorch-exported TTS model. onnx-ir now accepts both,
  matching ORT.
- **onnx-ir rejected non-power-of-two `n_fft`.** Upstream Burn's `stft`
  requires pow2 `n_fft` ([tracel-ai/burn#4865][burn-4865] tracks Bluestein's
  algorithm support). burn-onnx now emits a matrix-DFT path (compute
  twiddles at forward-call time, frame via `unfold`, matmul) when `n_fft`
  is non-pow2. The matmul runs in f64 internally and casts back to f32 to
  keep near-zero spectral components close to zero, so downstream
  `imag/real` phase recovery doesn't amplify f32 epsilon into huge spikes.
  Fast path unchanged.
- **LSTM initial state aliasing.** When `initial_h` and `initial_c` were
  the same ONNX tensor (e.g. both zero-constants), the codegen emitted
  `LstmState::new(x, x.clone())` where the leftmost bare use consumed `x`
  before `.clone()` could borrow it. The scope's clone tracker was correct;
  the LSTM codegen called `scope.arg` in the wrong order. Fixed.

[burn-4865]: https://github.com/tracel-ai/burn/issues/4865

## Usage

### 1. Download the model and generate a reference output

```bash
uv run get_model.py
# or: python get_model.py
```

This downloads `kokoro-v1.0.onnx` to the cache directory, extracts a node
summary (`node_info.json`), runs ONNX Runtime on a deterministic seeded test
case, and stores the waveform in `reference_outputs.json`.

### 2. Build and run the check

```bash
cargo build
cargo run
```

The check has two acceptance tiers:

- **Smoke tier (default).** Pass requires no NaN/Inf in the audio and Pearson
  r > 0.5 vs the ORT reference. This catches catastrophic regressions while
  tolerating the documented residual divergence
  ([burn-onnx#371][issue-371]) so a default `cargo run` does not exit(1) on
  expected behavior.
- **Strict tier (`KOKORO_STRICT=1`).** Same plus a 1%-of-peak tolerance on
  max-abs error and 0.1%-of-peak on mean-abs error. Use this once the
  residual divergence is fixed, to catch regressions:

  ```bash
  KOKORO_STRICT=1 cargo run --release
  ```

## Backend Support

```bash
cargo run                                          # flex (default)
cargo run --no-default-features --features wgpu    # WebGPU
cargo run --no-default-features --features metal   # Metal
```

## Test Input

The reference case uses a fixed seed (42) to produce 10 token IDs and a
scaled 256-d style vector. This is **not** a meaningful utterance — real use
requires phonemizing text with espeak-ng or similar — but it is sufficient
to verify that every op in the graph produces the same result under Burn as
it does under ORT.
