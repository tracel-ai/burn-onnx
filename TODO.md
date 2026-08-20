# burn-onnx roadmap

Prioritized work queue derived from a measured sweep of the open issues and the
`onnx-official-tests` scoreboard on 2026-08-18. Test counts come from re-running every non-passing
entry in `crates/onnx-official-tests/expectations.toml` through `onnx2burn`, then compile-checking
the output against `burn 0.22.0-pre.1` with the `flex` backend.

Items 1, 2 and 4 are done on this branch. Counts below are stated against the post-re-triage
baseline unless the text says otherwise.

## Scoreboard baseline

`expectations.toml` has 1615 entries. Current state, after the item 2 re-triage:

| Status         | Rows |
| -------------- | ---: |
| `pass`         |  811 |
| `fail-compare` |  216 |
| `skip-codegen` |  485 |
| `skip-compile` |  103 |

709 of those execute as harness tests. The rest are codegen-only: build.rs skips harness generation
for dynamic shapes, rank-0 I/O, and dtypes the `.pb` loader cannot construct.

### Why it had drifted

`build.rs` only verifies `pass` and `fail-compare` entries. `skip-codegen`, `skip-compile` and
`flaky` rows are read as documentation and never exercised, so they rot the moment someone fixes the
bug behind them, always in the pessimistic direction. Measured on `main` before this branch:

| Claimed            | Measured                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| 230 `skip-compile` | 192 codegen fine; 101 of those went on to pass                                                      |
| 230 `skip-compile` | 38 actually failed codegen (wrong status, not just wrong reason)                                     |
| 484 `skip-codegen` | 37 codegen fine (33 Mod-Shape, 4 QLinearMatMul)                                                     |
| 15 `skip-codegen`  | reason string stale: they now fail on training-domain ops, not the opset-domain check fixed in #434 |

`cargo xtask retriage` now re-checks every `skip-*` row, so this cannot silently recur.

### What "pass" does and does not mean

811 rows are marked `pass`; 705 of them execute as harness tests. The other 106 are codegen-only:
`build.rs` skips harness generation for dynamic shapes, rank-0 I/O, and dtypes the `.pb` loader
cannot construct, and `update-expectations` can only demote a row whose test failed. A codegen-only
row is therefore unfalsifiable once promoted, and its output is never compared against the
reference tensors. `retriage` now counts them separately when reporting promotions rather than
folding them into the total.

37 of this branch's 105 promotions are codegen-only, including `test_size` and `test_size_example`
(the Size fix is verified by the `crates/onnx-tests/tests/size/` integration tests, not by the
official suite) and 26 `test_castlike_*` rows converting to FLOAT8/INT4 variants. Extending the
harness to cover them is separate work; the honest reading of 811 is "811 compile, 705 match".

## Tier 1

### 1. Fix `Size` codegen

`crates/burn-onnx/src/burn/node/size.rs:17` emits:

```rust
let #output = #input.shape.num_elements();
```

`shape` is a method on `Tensor`, not a field, and `num_elements()` returns `usize` while the
generated signature declares `-> i64`. Every model containing an ONNX `Size` node produces Rust that
does not compile.

For a `Shape(N)` input it is worse: the generated code calls `.shape` on an `[i64; N]` array, which
has no such member. The answer there is the compile-time constant `N`.

Why it survived: the inline snapshot at `size.rs:38` asserts the broken output, and while
`tests/size/size.onnx` *is* registered in `crates/onnx-tests/build.rs`, `tests/test_mod.rs` never
declared `pub mod size;`. build.rs therefore generated the model on every build and nothing ever
`include!`d it, so rustc never saw the broken code. It is the only test directory in the repo
missing from `test_mod.rs` (`loop` and `mod` are there as raw idents).

Unblocks 21 official tests: `test_size`, `test_size_example`, and 19 `rms_normalization_*_expanded`.

**Status: done.** `size.rs` now branches on input type: Tensor -> `.shape().num_elements() as i64`,
`Shape(N)` -> the constant `N`, either scalar form -> 1. A new `Size(Shape(N)) -> constant N` rule
in `simplify/constant_shape.rs` folds the pattern away entirely and lets dead-node elimination drop
the feeding `Shape` node. Unlike the Gather and Slice rules it never consults `static_shape`, only
the rank, so it is safe under dynamic dimensions. `tests/size/` is wired up with the existing `size`
model plus a new `size_shape` model. Scoreboard moved 21 rows off `skip-compile`: 8 to `pass`, 13 to
`fail-compare` (see item 5).

### 2. Re-triage the scoreboard

Promote the clean and stale rows, correct the wrong reason strings, and demote the mislabeled
`skip-compile` rows to `skip-codegen`. Add a `cargo xtask retriage` that re-runs every `skip-*` row,
so the file cannot drift again.

**Status: done.** `cargo xtask retriage` runs codegen per row in its own process, promotes what
succeeds, then builds the test crate and demotes anything rustc rejects, carrying the diagnostic as
the row's reason. It converged on all 693 skipped rows in two compile rounds:

| Transition | Rows |
|---|---:|
| skip-compile -> pass | 101 |
| skip-compile -> skip-codegen (mislabeled) | 38 |
| skip-codegen -> skip-compile | 33 |
| skip-codegen -> pass | 4 |

`cargo xtask update-expectations` then demoted the 24 promotions whose output did not match the
reference tensors. Net against `main`:

| Status | Before | After |
|---|---:|---:|
| pass | 722 | 811 |
| fail-compare | 179 | 216 |
| skip-codegen | 484 | 485 |
| skip-compile | 230 | 103 |

Harness tests actually executing and passing went from 663 to 709. The reasons are worth more than
the counts: the single 207-row bucket reading "burn-onnx emits uncompilable generated code
(references alloc::\* from no_std, or emits unresolved variable bindings)" is gone, replaced by
exact rustc diagnostics. Item 8's clusters can now be read straight out of the file — the largest
being 34 rows of `expected Tensor<1, Bool>, found Tensor<1, Int>`, 22 of `expected f32, found
Tensor<0>`, and 22 of `expected f16, found f32`.

Two supporting fixes fell out of the sweep:

- retriage attributes errors in the generated `harness.rs` to the enclosing `fn`, not just errors in
  a generated model. Three `constantofshape` rows compile fine but the driver cannot call them: a
  Shape-typed graph input arrives as `[i64; N]` where the driver built a `Tensor<1, Int>`. Without
  the attribution the sweep aborted with "no error attributed to a promoted model".
- The `fail-compare` harness body in `onnx-official-tests/build.rs` now guards model construction
  with `catch_unwind`. `test_gru_batchwise` panics in `from_file` (the bpk is missing every
  `GateController` weight, which is its own bug worth a ticket), and that panic escaped the
  per-comparison guard and took down `verify_fail_compare_still_fails` for every other entry.

### 3. Upsample (#415)

A user is blocked importing a public model (`fastdepth_7.onnx`). Upsample is deprecated but common
in older exports, and is a strict subset of Resize (opset 7: `scales` attribute; opset 9: `scales`
input; modes nearest/linear). Currently a placeholder in
`crates/onnx-ir/src/node/unsupported.rs:94`.

### 4. Domain-aware unsupported-op error (#433)

`Unknown node type: VariantNotFound` for `TreeEnsembleRegressor` gives the user nothing to act on;
the reporter had to work out on their own that `ai.onnx.ml` is a separate domain.

**Status: already fixed on main**, after the 0.21.0 release the issue was filed against.
`proto_conversion.rs` now maps any unrecognised standard-domain op to `NodeType::Custom` rather
than unwrapping a `FromStr`, and the custom-op coverage check reports it by domain. Checked with a
hand-built `ai.onnx.ml::TreeEnsembleRegressor` model:

```
INFO onnx_ir::proto_conversion: Custom-domain op 'ai.onnx.ml::TreeEnsembleRegressor'
  (node 'tree1'); treating as custom op
...
model contains 1 custom op(s) with no covering inference hook:
  - ai.onnx.ml::TreeEnsembleRegressor used by 1 node(s)
Register hooks via ModelGen::register_custom_op.
```

Remaining work is issue hygiene: confirm against the reporter's attached model, then close #433
against the next release and fold the operator request into #162.

## Tier 2

### 5. Reduce family comparison failures (99 tests)

The largest `fail-compare` bucket, and the worst failure mode to leave sitting: these compile and
run, and produce wrong numbers. The re-triage grew this from 88 to 99 by promoting reduce rows that
turned out to compile and then miscompare.

| Family      | Tests |
| ----------- | ----: |
| reduce_sum  |    24 |
| reduce_log  |    18 |
| reduce_l1   |    14 |
| reduce_l2   |    14 |
| reduce_max  |     8 |
| reduce_min  |     8 |
| reduce_prod |     7 |
| reduce_mean |     6 |

Likely a shared root cause in keepdims / empty-axes / `noop_with_empty_axes`.

Item 1 produced a sharp reproducer for part of this. Once the 19 `rms_normalization_*_expanded`
models compile, exactly 6 pass: the `axis0` and `axis_negative_<rank>` variants. Every other axis is
off by a uniform relative factor. The generated code is

```rust
let reducemean1_out1 = { mul1_out1.mean().expand([1; 2usize]) };
```

`mean()` with no argument reduces over *all* elements, so it is only correct when `axis` names the
first dimension and the reduction therefore covers the whole tensor. It should reduce over the axes
from `axis` onward. Those 13 are now tracked as `fail-compare` against #311.

Filed as **#459**: opset 18 moved `axes` from an attribute to an input, and a runtime `axes` input
leaves `ReduceConfig::dims` empty (`onnx-ir/src/node/reduce.rs:275`), which burn-onnx cannot tell
apart from "no axes given, reduce everything" (`burn-onnx/src/burn/node/reduce.rs:81`). Whether the
rest of this 99-row bucket shares that cause is unverified, but it is the cheapest confirmed way in.
`noop_with_empty_axes` is a third unhandled meaning for empty axes and should be settled in the same
fix.

### 6. Runtime weight inputs: LayerNorm (#352, 19 tests) + Conv/ConvTranspose (#346, 12 tests)

Both are the same fix: route through the functional API (`burn::tensor::module::conv2d`) instead of
a baked-in `Param` field. Five ops have now hit this pattern; extract the shared
`runtime_scalar_to_native(arg, target_dtype, scope)` helper in `argument_helpers.rs` proposed in the
#314 thread before doing these two.

### 7. RMSNormalization (19 tests, +19 more via item 1)

Burn has `RmsNorm` natively and ONNX 23 made this a first-class op. High real-world relevance:
Llama, Qwen and Gemma all use it.

### 8. Remaining compile-error clusters

All 103 remaining `skip-compile` rows now carry the rustc diagnostic that produced them, so this
table is a `grep` of `expectations.toml` rather than an estimate. Sorted by blast radius:

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

### 9. GRU/LSTM/RNN discard runtime weights (#458)

Surfaced by the item 2 sweep, previously hidden behind a `skip-compile` row. `test_gru_batchwise`
compiles, then panics in `Model::from_file`:

```
Validation error: Missing tensors: [
  ("gru1.new_gate.hidden_transform.weight", "Struct:Model.Struct:Gru.Struct:GateController.Struct:Linear"),
  ("gru1.new_gate.input_transform.weight",  ...),
  ("gru1.reset_gate.hidden_transform.weight", ...),
  ("gru1.reset_gate.input_transform.weight",  ...),
  ("gru1.update_gate.hidden_transform.weight", ...),
  ("gru1.update_gate.input_transform.weight",  ...),
]
```

Root-caused while filing #458, and it is worse than "unloadable". Every RNN-family test in the
upstream suite supplies `W`/`R` as runtime graph inputs rather than initializers.
`collect_gru_snapshots` returns an empty snapshot list when the weights are not statically available
(`gru.rs:52`, `:55`; same shape in `lstm.rs:87`/`:90` and `rnn.rs:82`/`:85`), but `field()` still
emits the module. So the generated `forward` accepts `w` and `r` as parameters and never reads them:

```rust
pub fn forward(&self, x: Tensor<3>, w: Tensor<3>, r: Tensor<3>) -> Tensor<3> {
    let gru_output = self.gru1.forward(x.swap_dims(0, 1), None);
    //                    ^^^^ w and r are dropped on the floor
```

`from_file` panics on the missing tensors, but `Model::new` does not: `GruConfig::init` gives the
module fully random weights and inference proceeds. That silent path is the reason this is a bug
rather than a gap. Same family as items 6's Conv/LayerNorm runtime weights (#346, #352), except
those reject the model with a clear error instead of accepting it and computing nonsense.

## Tier 3

- **#50 / #51 metal backend.** Dozens of ops fail and YOLO11x diverges by 295 max-abs. Correctness
  on the backend people ship on outweighs op-count wins. Root cause probably belongs upstream in
  burn, but it surfaces here.
- **#280 shape propagation through Where/Mul/ConstantOfShape.** Blocks RF-DETR without an `onnxsim`
  pre-pass.
- **#371 Kokoro residual 1.3x.** Established as f32 drift through HiFi-GAN resblocks, not fixable
  here. Close or move to burn.

## Deprioritized

- **NegativeLogLikelihoodLoss (52) + SoftmaxCrossEntropyLoss (34) + nllloss fail-compare (12).** 98
  tests, but training-loss ops in an inference-focused importer. Large count, small user value.
- **#433 TreeEnsembleRegressor / #162 ONNX-ML.** The reporter reached the right conclusion
  themselves, and the error message they hit is already fixed (item 4). Close #433 against the next
  release; the operator request itself belongs in #162.
- **Float8 / Float4 / INT4 cast tests (~30).** Blocked on backend dtype support, not on burn-onnx.

## Order

Items 1 and 2 are done; 4 turned out to be already fixed on `main`. Item 3 is next.

Then 5 -> 6 -> 7, each now measurable against an honest baseline. Items 5 and 9 now have issues (#459, #458); both are
silent-wrong-answer bugs, which argues for pulling them ahead of the pure test-count work. #460
(non-deterministic attribute errors) should land early too: it is small, and until it does, every
`retriage` run churns a couple of rows for no reason.

Item 8's top three rows (34 + 22 + 22 = 78 of the 103 remaining `skip-compile` rows) are probably
two fixes, which makes that bucket competitive with item 5 on effort-per-test.
