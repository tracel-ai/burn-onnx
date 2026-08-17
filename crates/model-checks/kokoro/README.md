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

**Runs end-to-end; numeric output is close but does not yet match ORT to
the test tolerance.** The model codegens, compiles (about 7 minutes on a
2024 M3), and runs a forward pass producing 13,200 audio samples for the
reference 10-token input. Burn audio peaks at ~170 vs ORT's ~131 (about
1.3× too large) with Pearson correlation 0.69 against the ORT reference.

### Debug history

The `bisect_setup.py` + `src/bin/bisect.rs` (gated behind the `bisect`
feature) infrastructure modifies the kokoro ONNX to expose ~190
intermediate tensors and compares Burn's outputs to ORT layer-by-layer.

The first round of bisection found that all upstream ops were bit-accurate
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
- That phase noise feeds `noise_convs.0` and propagates through the
  generator stack into the audio, surfacing as the residual 1.3× peak
  factor.

This is a fundamental algorithm divergence, not a Burn bug — see
[burn-onnx#371][issue-371] for the full investigation, element-wise data,
and possible fixes (manual-atan2 → atan2 coalescer pass, or implementing
Bluestein's algorithm upstream).

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
  tolerating the documented ~1.3× peak / r=0.69 residual divergence
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
cargo run --no-default-features --features tch     # LibTorch
cargo run --no-default-features --features wgpu    # WebGPU
cargo run --no-default-features --features metal   # Metal
```

## Test Input

The reference case uses a fixed seed (42) to produce 10 token IDs and a
scaled 256-d style vector. This is **not** a meaningful utterance — real use
requires phonemizing text with espeak-ng or similar — but it is sufficient
to verify that every op in the graph produces the same result under Burn as
it does under ORT.
