# burn-onnx roadmap

Prioritized work queue derived from a measured sweep of the open issues and the
`onnx-official-tests` scoreboard on 2026-08-18. Test counts come from re-running every non-passing
entry in `crates/onnx-official-tests/expectations.toml` through `onnx2burn`, then compile-checking
the output against `burn 0.22.0-pre.3` with the `flex` backend.

Three items have shipped since that sweep (#461, #464, #466); "Landed" at the bottom says what each
one did, and `git log -- TODO.md` has the write-ups they replaced. Findings from those items that
describe work still open were carried into Tier 2. Counts below are the current state of
`expectations.toml` and already include all three.

## Scoreboard baseline

`expectations.toml` has 1615 entries:

| Status         | Rows |
| -------------- | ---: |
| `pass`         |  932 |
| `fail-compare` |   96 |
| `skip-codegen` |  484 |
| `skip-compile` |  103 |

830 of the 932 `pass` rows execute as harness tests. The other 102 are codegen-only: build.rs skips
harness generation for dynamic shapes, rank-0 I/O, and dtypes the `.pb` loader cannot construct.

### Why skip counts rot

`build.rs` only verifies `pass` and `fail-compare` entries. `skip-codegen`, `skip-compile` and
`flaky` rows are read as documentation and never exercised, so they go stale the moment someone
fixes the bug behind them, always in the pessimistic direction. Measured on `main` before #457: of
230 claimed `skip-compile` rows, 192 built fine and 101 of those went on to pass, while 38 were not
compile failures at all but codegen failures wearing the wrong status.

`cargo xtask retriage` now re-checks every `skip-*` row, so this cannot silently recur — but nothing
runs it automatically. Run it before trusting a skip count, and after any fix that could plausibly
unblock a family.

### What "pass" does and does not mean

932 rows are marked `pass`; 830 of them execute as harness tests. The other 102 are codegen-only:
`build.rs` skips harness generation for dynamic shapes, rank-0 I/O, and dtypes the `.pb` loader
cannot construct, and `update-expectations` can only demote a row whose test failed. A codegen-only
row is therefore unfalsifiable once promoted, and its output is never compared against the
reference tensors. `retriage` counts them separately when reporting promotions rather than folding
them into the total.

`test_size` and `test_size_example` are in that group (the Size fix is verified by the
`crates/onnx-tests/tests/size/` integration tests, not by the official suite), as are 26
`test_castlike_*` rows converting to FLOAT8/INT4 variants. Extending the harness to cover them is
separate work; the honest reading of 932 is "932 compile, 830 match".

#464 turned four of those unfalsifiable rows into real tests and immediately found a bug in two of
them, which is the concrete cost of the category: `test_reduce_log_sum_exp_do_not_keepdims_*` were
marked `pass` while producing a rank-0 output the harness declines to drive. Correcting the inferred
rank gave them a driver, and the driver failed.

## Tier 1

### 1. Runtime weight inputs: LayerNorm (#352, 19 tests) + Conv/ConvTranspose (#346, 12 tests)

Both are the same fix: route through the functional API (`burn::tensor::module::conv2d`) instead of
a baked-in `Param` field. Five ops have now hit this pattern; extract the shared
`runtime_scalar_to_native(arg, target_dtype, scope)` helper in `argument_helpers.rs` proposed in the
#314 thread before doing these two.

Unlike the RNN family (#466), these three do have a functional route: `burn-tensor` exports
`conv1d`/`conv2d`/`conv3d`, `conv_transpose*` and `layer_norm` as free functions taking the weights
as arguments, so no module has to be built per forward call.

### 2. Remaining compile-error clusters (103 rows)

All 103 remaining `skip-compile` rows carry the rustc diagnostic that produced them, so this table
is a `grep` of `expectations.toml` rather than an estimate. Sorted by blast radius:

| Rows | Diagnostic | Example | Read |
| ---: | --- | --- | --- |
| 34 | `expected Tensor<1, Bool>, found Tensor<1, Int>` | `test_attention_3d_attn_mask_expanded` | the Mod/And-on-Shape chain lands an Int tensor where a mask is wanted. Biggest single win left in the bucket, and it is the attention-expanded family. |
| 22 | `expected f32, found Tensor<0>` | `test_cast_FLOAT8E4M3FN_to_FLOAT` | rank-0 output typed as a scalar in the signature but produced as a tensor. |
| 22 | `expected f16, found f32` | `test_cast_FLOAT8E4M3FN_to_FLOAT16` | the f16 cast result is never narrowed. Same family as the row above; likely one fix for both 44. |
| 3 | `no method named add found for type f32` | `test_blackmanwindow_expanded` | scalar/tensor mixing in the window ops. |
| 3 | `expected Tensor<1>, found f32` | `test_hammingwindow_symmetric_expanded` | same family. |
| 3 | `use of moved value: div1_out1` | `test_dynamicquantizelinear_expanded` | clone tracking missed; `arg_to_ident` used where `scope.arg` was needed. |
| 2 | `expected bool, found Tensor<0, Bool>` | `test_equal_string` | rank-0 bool, same shape as the f32 case above. |
| 2 | `Tensor<3>: ElementConversion is not satisfied` | `test_gelu_default_2_expanded` | a tensor passed to a scalar-taking API. |
| 2 | `cannot find value maxpool2d1_out2` | `test_maxpool_with_argmax_2d_precomputed_pads` | MaxPool's second output is never bound. |
| 2 | `expected Tensor<1>, found Tensor<1, Int>` | `test_pow_types_float32_int32` | missing cast before a binary op. |
| 2 | `no method named powf on Tensor<Int>` | `test_pow_types_int32_float32` | Pow with an int base needs a cast first. |
| 3 | `expected [i64; N], found Tensor<1, Int>` | `test_constantofshape_float_ones` | not a codegen bug: the model compiles, the generated *harness* cannot call it, because a Shape-typed graph input arrives as `[i64; N]` where the driver built a `Tensor<1, Int>`. Fixing it means teaching build.rs to read the generated `forward` signature rather than inferring argument types from the ONNX proto. |
| 1 each | `fmod` on `Tensor<Int>`, `expected Tensor<4>, found Tensor<4, Int>`, `can't compare f32 with {integer}` | `test_mod_int64_fmod` | one-offs. |

The `half::f16` problem noted during the first sweep is not in this table: generated signatures do
name `half::f16`, but `onnx-official-tests` happens to depend on `half` already. It still bites a
consuming crate that does not, and is worth fixing independently of these rows.

### 3. RMSNormalization (19 tests)

Burn has `RmsNorm` natively and ONNX 23 made this a first-class op. High real-world relevance:
Llama, Qwen and Gemma all use it. The 19 `_expanded` rows this item used to also claim now pass on
the decomposition alone (#464); the 19 left are the native op, still `skip-codegen`.

## Tier 2

- **#50 / #51 metal backend.** Dozens of ops fail and YOLO11x diverges by 295 max-abs. Correctness
  on the backend people ship on outweighs op-count wins. Root cause probably belongs upstream in
  burn, but it surfaces here.
- **Resize shares every gap Upsample closed in #461 (surfaced by that review).** None of it is
  new and none of it blocked #461, but it is all in `crates/burn-onnx/src/import/burn/node/resize.rs`
  and its processor:
  - Accepts `asymmetric` linear and computes half-pixel values (#311 already tracks
    `test_resize_upsample_scales_cubic_asymmetric` as `fail-compare`).
  - Does not validate that a nearest scale divides its dimension, so it has the same silent pixel
    shift Upsample now refuses.
  - The runtime path emits `input_dims[3]` unconditionally, so a rank-3 runtime-scales Resize emits
    code that does not compile.
  - Drops `scales[0]`/`scales[1]` in the runtime path, so a batch or channel scale that the static
    path hard-rejects is silently ignored when the same tensor arrives as a graph input.
  - Leaves `nearest_mode` at the opset 11 default on its opset 10 path, where the spec says floor.

  Note `coordinate_transformation_mode` and `nearest_mode` are recorded but never read by codegen
  (only `align_corners` is derived), so pinning them documents intent without changing behavior.
- **`build_node` cannot report an error, so late-lifted constants panic.** `lift_constants` runs
  again after identity elimination (`post_processing.rs:265`) and type inference does not re-run
  after it, so `Constant -> Identity -> Op` reaches `build_node` with a value that was Dynamic
  during `infer_types`. Any validation that first becomes possible there can only panic:
  `NodeProcessor::build_node` returns `Node`, not `Result<Node>`. Every processor in the crate has
  this shape (`.expect("Config extraction failed")`); Upsample and Reduce are the first two with
  checks that can realistically fire there - an out-of-range axis behind a
  `Constant -> Identity -> Reduce` chain is the second case. Reproduced with a
  `Constant -> Identity -> Upsample` graph carrying scales of 1.75.
- **The static-vs-runtime weight decision is stated on both sides of the crate boundary (surfaced
  by the #466 review).** `lift_all_or_none`'s liftability test in onnx-ir and `weights_are_runtime`
  in burn-onnx encode the same invariant with two non-complementary predicates over a four-variant
  `ValueSource`, kept in sync by hand. `BatchNormalizationNode` already shows the deeper answer:
  decide once in `extract_config` and carry it in the IR config (`BatchNormConfig::Static |
  Runtime`), so codegen matches rather than re-derives. Doing the same for GRU/LSTM/RNN churns every
  positional `Config::new(...)` in their tests, which is why it was left out of #466.
- **Generated `use` lines are predicted per node.** `BurnImports` emits bare `use` with no
  `#[allow(unused_imports)]`, so LSTM and RNN carry a `needs_module_type` flag purely to avoid
  importing a type the runtime path never names. GRU needs none of it because it uses fully-qualified
  `burn::nn::gru::Gru` paths. Either adopt GRU's style in the other two, or allow unused imports in
  the generated block once.
- **#280 shape propagation through Where/Mul/ConstantOfShape.** Blocks RF-DETR without an `onnxsim`
  pre-pass.
- **#371 Kokoro residual 1.3x.** Established as f32 drift through HiFi-GAN resblocks, not fixable
  here. Close or move to burn.
- **Empty-set reductions return the wrong identity (3 rows, left over from #464).**
  `test_reduce_max_empty_set`, `test_reduce_min_empty_set` and `test_reduce_log_sum_exp_empty_set`
  reduce over a zero-size dimension, where ONNX mandates the identity element (`-inf` for max,
  `+inf` for min) and Burn's kernels return something else. Sum, prod, L1, L2 and LogSum over an
  empty set all pass, because their identities are 0 and 1 and Burn agrees. Belongs upstream in burn.
- **Two latent RNN weight bugs on the static path (out of scope for #466, both predate it).**
  `collect_*_snapshots` slices a direction with `.squeeze::<2>()`, which drops *every* size-1
  dimension, so a GRU with `input_size == 1` or `hidden_size == 1` squeezes the wrong axis and fails
  the rank check; the generated runtime path uses `select_dim`/`slice_dim` and is not exposed. And
  nothing validates a declared `W`/`R` shape against the `hidden_size` and `direction` attributes:
  an undersized weight panics in Burn's slice check, an oversized one is silently truncated, so a
  `[2, ...]` W in a `direction="forward"` model quietly uses direction 0 only. That check belongs in
  `infer_types` as a `ProcessError` where the static shape is known.

## Deprioritized

- **NegativeLogLikelihoodLoss (52) + SoftmaxCrossEntropyLoss (34) + nllloss fail-compare (12).** 98
  tests, but training-loss ops in an inference-focused importer. Large count, small user value.
- **#433 TreeEnsembleRegressor / #162 ONNX-ML.** The reporter reached the right conclusion
  themselves, and the error message they hit is already fixed on `main`: `proto_conversion.rs` maps
  any unrecognised standard-domain op to `NodeType::Custom` instead of unwrapping a `FromStr`, and
  the custom-op coverage check reports it by domain. Remaining work is issue hygiene: confirm
  against the reporter's attached model, close #433 against the next release, and fold the operator
  request into #162.
- **Float8 / Float4 / INT4 cast tests (~30).** Blocked on backend dtype support, not on burn-onnx.

## Order

Item 1 first. Conv, ConvTranspose and LayerNorm all have a functional entry point in `burn-tensor`
to route through, so the mechanism is settled before the work starts, and it is 31 rows across two
issues. #466 did not produce a helper item 1 can reuse - the two share a symptom, not a mechanism -
so it starts from the `runtime_scalar_to_native` extraction proposed in the #314 thread, not from
`node/rnn_common.rs`.

Item 2 is the bigger count (its top three rows are 78 of the 103 remaining `skip-compile` rows, and
are probably two fixes) but the table there is a set of rustc diagnostics, not a diagnosis. Then 3.

Run `cargo xtask retriage` before trusting any count in this file.

## Landed

Removed from the queue above. `git log -- TODO.md` has the full write-ups.

- **#461 Upsample (#415).** Lowered onto `ResizeNode` with `asymmetric` coordinates and `floor`
  rounding pinned, so there is no second copy of the interpolate codegen. Refuses linear mode and
  nearest scales that do not divide their dimension rather than shifting pixels silently. Resize's
  copy of the same gaps is in Tier 2.
- **#464 Reduce runtime `axes`, plus #460 non-deterministic attribute errors.** 109 rows promoted
  from `fail-compare`, harness tests 706 -> 819. `ReduceConfig::axes` is now `Static | Runtime`, so
  "no axes given", `noop_with_empty_axes` and "axes arrive at run time" stop sharing an empty `Vec`.
  Only the output rank has to be static: Burn's `*_dims` take a runtime slice. The 3 empty-set rows
  it could not fix are in Tier 2.
- **#466 GRU/LSTM/RNN runtime weights (#458).** 11 rows promoted, harness tests 819 -> 830. Burn
  exports no functional `gru`/`lstm`/`rnn`, so the generated `forward` builds the module and
  overwrites each gate's `Param` from slices of the runtime tensors, table-driven from one
  `GateLayout` const per op. Also fixed `layout=1` putting the direction axis on the wrong output
  dimension. Two latent static-path bugs it left alone are in Tier 2.
