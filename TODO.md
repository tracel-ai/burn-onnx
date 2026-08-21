# burn-onnx roadmap

Prioritized work queue derived from a measured sweep of the open issues and the
`onnx-official-tests` scoreboard on 2026-08-18. Test counts come from re-running every non-passing
entry in `crates/onnx-official-tests/expectations.toml` through `onnx2burn`, then compile-checking
the output against `burn 0.22.0-pre.1` with the `flex` backend.

The `Size` codegen fix and the scoreboard re-triage that produced this baseline landed in #457, and
the domain-aware unsupported-op error (#433) turned out to be already fixed on `main`. Counts below
are the current state of `expectations.toml`, including the Upsample promotion (item 1), the
runtime-`axes` reduce fix (item 2) and the RNN-family runtime weights fix (item 6).

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

Item 2 turned four of those unfalsifiable rows into real tests and immediately found a bug in two
of them, which is the concrete cost of the category: `test_reduce_log_sum_exp_do_not_keepdims_*`
were marked `pass` while producing a rank-0 output the harness declines to drive. Correcting the
inferred rank gave them a driver, and the driver failed.

## Tier 1

### 1. Upsample (#415)

A user is blocked importing a public model (`fastdepth_7.onnx`). Upsample is deprecated but common
in older exports, and is a strict subset of Resize (opset 7: `scales` attribute; opset 9: `scales`
input; modes nearest/linear).

**Status: done.** ONNX deprecated Upsample *into* Resize: Resize opset 10 is Upsample with the
coordinate mapping (`asymmetric`) and nearest rounding (`floor`) spelled out as attributes. So
`node/upsample.rs` extracts Upsample's own attributes across all three shapes it has had -
`height_scale`/`width_scale` (opset 1), the `scales` float-list attribute (opset 7), the `scales`
input (opset 9+, static or runtime) - and builds a `ResizeNode` with those two modes pinned. The
enum entry is `Upsample => resize::ResizeNode` and burn-onnx just lists `Upsample` in its dispatch
macro, so there is no second copy of the interpolate codegen. Node naming still comes from the
`RawNode`, so generated code reads `self.upsample1`, not `resize1`.

Beyond what Resize does today, the processor computes the output static shape as
`floor(dim * scale)` rather than copying the input's, rejects scaling the batch or channel
dimension instead of silently dropping those scales, and rejects ranks Burn's interpolate cannot
serve (anything but 3 and 4, or anything but 4 when the scales are a runtime input) rather than
emitting code that references a field that was never created.

A multi-agent review caught a wrong claim in the first draft of this work, which said nearest mode
was exact and only linear diverged. Both halves were wrong, and the corrected behavior is the more
interesting part of the change:

- **`mode="linear"` is refused**, not warned about. Burn's bilinear samples at half-pixel
  coordinates (ONNX's `half_pixel`); Upsample mandates `asymmetric`. Every interior sample differs
  at every scale other than 1, so this is "always wrong", not "may differ". A `log::warn!` was also
  the wrong channel: cargo swallows build-script stdout unless the line is `cargo:warning=`
  prefixed, so the primary `build.rs` path showed the user nothing at all.
- **Nearest is refused when a scale does not divide its dimension evenly.** ONNX picks a source
  element by scale, `floor(o / scale)`; Burn's kernel picks it by output size,
  `floor(o * in / out)` with `out = floor(in * scale)`. Those agree only when `in * scale` is
  whole. Verified end to end: `scale=1.75` on width 5 gives Burn `[0,0,1,1,2,3,3,4]` against
  onnxruntime's `[0,0,1,1,2,2,3,4]`. Where the product is provable the model is rejected with the
  dimension, the scale and the reason; where it is not (runtime scales, dynamic dims) it warns.
  Integer scales, which is what fastdepth and most real exports use, are unaffected.

Every test in the first draft used an integer scale, which is exactly the case where the two
formulas coincide, so the suite could not have caught this. `ReferenceEvaluator` cannot either: its
Upsample is `np.repeat` and raises on non-integer scales, making onnxruntime the only usable oracle.

Two smaller review fixes: opset 1 spells linear mode `bilinear` (the rename came at opset 7), which
was being rejected as an unknown mode rather than for the real reason; and spatial scales are now
checked against the spec's "greater than or equal to 1", since a scale below 1 or a NaN reaches
`as usize` in generated code and saturates to a zero-size dimension.

Scoreboard: `test_upsample_nearest` moved from `skip-codegen` to `pass`. Opset compliance grew from
472 to 476 op-version combinations (Upsample at opsets 1, 7, 9, 10).

## Tier 2

### 2. Reduce family comparison failures (99 tests)

**Status: done (#459).** 96 of the 99 rows shared one root cause, and it was the one #459 named. Opset 18
moved `axes` from an attribute to an input; when that input is a graph input rather than a constant
its value is unknown at build time, and `ReduceConfig::dims` was a plain `Vec<usize>` that recorded
that case as an empty vector — the same value ONNX uses for "no axes given, reduce everything".
Codegen then emitted `.sum()` / `.mean()` with no dimension argument and dropped the axes input on
the floor. Every one of those models supplies `axes` this way; none of them was a keepdims or
broadcasting bug. The other 3 are a separate bug, described at the end of this item.

The fix is to make the three meanings of "empty axes" distinguishable, which they are not in a
`Vec`:

| ONNX                                  | `ReduceConfig::axes`  | Behavior                |
| ------------------------------------- | --------------------- | ----------------------- |
| `axes` absent, or an empty list       | `Static(vec![])`      | reduce every axis       |
| empty list with `noop_with_empty_axes`| `Static(vec![])`      | skip the reduction      |
| `axes` supplied at run time           | `Runtime(input_ref)`  | reduce what it names    |

`ReduceConfig` now carries `axes: ReduceAxes`, the `Static(..) | Runtime(RuntimeInputRef)` enum that
22 other node files already use, plus the `noop_with_empty_axes` attribute it was previously
discarding. "Skip the reduction" is not quite "identity": the spec says other operations still
happen, so `ReduceSumSquare` still squares and `ReduceL1` still takes an absolute value.

Reducing over axes that are not compile-time constants sounds like it should be impossible against
Burn's statically-ranked tensors, and it very nearly is — but only the output *rank* has to be
static, not the axis values. Burn 0.22's `sum_dims`/`mean_dims`/`max_dims`/`min_dims`/`prod_dims`
and `squeeze_dims::<D2>` all take `&[impl AsIndex]`, a runtime slice, and `AsIndex::try_dim_index`
wraps negative entries itself, so a runtime axis and a negative axis both come out correct with no
work in the generated code. The rank comes from two places: with `keepdims=1` it is the input rank,
and with `keepdims=0` it is `input_rank - len(axes)`, where the *length* is in the axes input's
static shape even when its values are not. All 99 models declare that length. When they do not
(a `Range`-computed axes list) and `keepdims` is off, onnx-ir now refuses the model instead of
guessing.

Generated code for a runtime-axes reduce reads the axes once and hands the slice to Burn:

```rust
let __axes: alloc::vec::Vec<i64> = axes.into_data().iter::<i64>().collect();
data.abs().sum_dims(&__axes).squeeze_dims::<2usize>(&__axes)
```

Two things surfaced while doing it that were not in the original diagnosis:

- **LogSumExp was wrong for `keepdims=0` independently of the axes bug.** It subtracts a running
  max from the input, so that max has to keep its rank to broadcast back; squeezing it first made
  `expand(input_shape)` either wrong or a hard `Squeeze` panic. Both intermediate reductions now run
  with keepdims and the reduced axes are dropped once at the end. This was invisible before because
  the two affected rows inferred a rank-0 output, which `build.rs` declines to generate a driver for
  — they were marked `pass` and never ran.
- **Out-of-range axes were accepted.** `dim as usize` on a negative that did not wrap left a huge
  index; `extract_config` now returns `ProcessError::InvalidAttribute` naming the axis and the rank.

Scoreboard: 109 rows promoted from `fail-compare` to `pass` — 96 of the 99 reduce rows and all 13
remaining `rms_normalization_*_expanded` rows, which used `Shape -> Size -> Range` to build their
axes at run time. Harness tests went from 706 to 819, all green.

A multi-agent review after the fact turned up two more silent-wrong-answer cases in the first
draft of this work, both now fixed and both worth recording because they are the same shape as the
bug being fixed:

- **`noop_with_empty_axes` was lowered to a bare identity.** The spec says the reduction is skipped
  but "other operations will be performed", so `ReduceSumSquare` must still square, `ReduceL1` and
  `ReduceL2` must still take an absolute value, and `ReduceLogSum` must still take a log. Only the
  five plain reductions are genuine identities. Modelling this as "reduce over no axes" rather than
  an early return makes each composite land on the right answer through machinery that already
  exists - `ReduceL2` becomes `sqrt(square(x))`, `ReduceLogSumExp` becomes `x + log(exp(x - x))`.
  The reductions that *are* identities now implement `NodeProcessor::is_noop`, so the framework
  drops those nodes in post-processing instead of codegen emitting a rebinding; the codegen path
  stays for `simplify(false)`, where only Identity is eliminated.
- **A runtime axes list that is empty at run time.** Burn's `*_dims` fold over an empty slice is the
  identity, but ONNX reads empty axes as "every dimension" unless `noop_with_empty_axes` is set.
  Only reachable when the axes input has no statically known length, which is exactly the case the
  opset 18 input shape exists for. The generated code now resolves the list where its length is
  finally known.

Two structural validations were added alongside: an axis count larger than the input rank used to
underflow `tensor_rank - axis_count` and panic with no node name, and duplicate axes built cleanly
and then panicked inside Burn's `squeeze_dims`, which deduplicates and so disagreed with the rank
onnx-ir had declared.

Note for the `build_node` item in Tier 3: Reduce is now the second operator, after Upsample, with
validation that can first become reachable in `build_node` (an out-of-range axis behind a
`Constant -> Identity -> Reduce` chain). Its panic at least names the node and formats the error
with `Display` now, but the underlying hazard is unchanged.

The 3 reduce rows left are `test_reduce_max_empty_set`, `test_reduce_min_empty_set` and
`test_reduce_log_sum_exp_empty_set`, which are a different bug: reducing over a zero-size dimension,
where ONNX mandates the identity element (`-inf` for max, `+inf` for min) and Burn's kernels return
something else. Sum, prod, L1, L2 and LogSum over an empty set all pass, because their identity
elements are 0 and 1 and Burn agrees. This belongs upstream in burn, not here.

Also fixed in the same pass, since it was blocking clean `retriage` runs: **#460**, non-deterministic
attribute-validation errors. `Attributes` was a `HashMap`, whose iteration order Rust reseeds per
process, so a model with two rejected attributes reported whichever one the loop happened to reach
first. It is now a `BTreeMap`; the type change is one line and the fallout was the 5 construction
sites the issue predicted plus their test helpers. `test_resize_downsample_sizes_nearest_not_smaller`
reported `axes` on 8 of 8 runs afterwards, against 1-of-6 before.

### 3. Runtime weight inputs: LayerNorm (#352, 19 tests) + Conv/ConvTranspose (#346, 12 tests)

Both are the same fix: route through the functional API (`burn::tensor::module::conv2d`) instead of
a baked-in `Param` field. Five ops have now hit this pattern; extract the shared
`runtime_scalar_to_native(arg, target_dtype, scope)` helper in `argument_helpers.rs` proposed in the
#314 thread before doing these two.

Unlike the RNN family (item 6), these three do have a functional route: `burn-tensor` exports
`conv1d`/`conv2d`/`conv3d`, `conv_transpose*` and `layer_norm` as free functions taking the weights
as arguments, so no module has to be built per forward call.

### 4. RMSNormalization (19 tests)

Burn has `RmsNorm` natively and ONNX 23 made this a first-class op. High real-world relevance:
Llama, Qwen and Gemma all use it. The 19 `_expanded` rows this item used to also claim now pass on
the decomposition alone (item 2); the 19 left are the native op, still `skip-codegen`.

### 5. Remaining compile-error clusters

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

### 6. GRU/LSTM/RNN discard runtime weights (#458)

**Status: done.** All three ops accepted models whose `W`/`R` arrive as runtime graph inputs and
then silently discarded them: each `collect_*_snapshots` returned an empty snapshot list from its
two `let Some(..) else { return vec![] }` weight guards when the weights were not statically
available, while `field()` emitted the module regardless. The generated `forward` took the
weight tensors as parameters and never read them. `Model::from_file` panicked on the missing
tensors; `Model::new` did not, and ran inference on `GruConfig::init`'s random weights.

The issue offered two fixes: consume the runtime weights, or reject the model with a clear error.
The second was the smaller change, but it would have made every RNN test in the upstream suite
`skip-codegen` - the category that, per "Why skip counts rot" above, nothing ever re-checks. The
weights are now consumed.

**The functional route this item was expected to share with item 3 does not exist.**
`burn-tensor/src/tensor/module.rs` exports functional `conv1d`/`conv2d`/`conv3d`,
`conv_transpose*`, `layer_norm` and `linear`, which is exactly what #346 and #352 need. It exports
no `gru`, `lstm` or `rnn`. So the earlier guess that doing 6 first would produce the helper 3 needs
was wrong; these are two different fixes that happen to share a symptom.

What works instead is that Burn's recurrent modules expose their parameters. When the weights are
runtime, `field()` returns `None` - no struct field, so nothing for the snapshot pipeline to fail to
fill - and `forward` builds the module locally, then overwrites each gate's `Param` from slices of
the runtime tensors:

```rust
let mut gru1 = burn::nn::gru::GruConfig::new(2, 5, false)
    .with_reset_after(false)
    .init(&self.device);
let __w_dir = __w.select_dim::<2>(0, 0);
gru1.update_gate.input_transform.weight = burn::module::Param::from_tensor(
    __w_dir.clone().slice_dim(0, 0..5).transpose(),
);
```

This is the same slice-and-transpose `collect_*_snapshots` already runs at build time, emitted as
tokens instead. It is also not a new coupling to Burn's internals: burn-onnx already depended on
`gru1.update_gate.input_transform.weight`, as a string, in the snapshot paths. Field access trades a
load-time `Missing tensors` failure for a compile error if Burn ever renames one.

A first draft passed `Initializer::Zeros` here on the theory that the config's Xavier draw was
wasted work. It is not: `Initializer::init_with` returns `Param::uninitialized`, whose closure runs
only on the first `val()`, and every parameter is replaced before anything reads one. The config
allocates no tensor data either way, so the initializer was three lines of emitted code buying
nothing.

The three ops differ in only three constants, so one table-driven emitter in
`burn-onnx/src/import/burn/node/rnn_common.rs` covers them. The `GateLayout` const drives each op's
build-time `collect_*_snapshots` as well, so the gate mapping has one home; its `BiasLayout` drives
the emitter only, and each collector still hardcodes the matching bias policy by hand:

| Op   | Burn gate order             | ONNX gate index | `B` handling                        |
| ---- | --------------------------- | --------------- | ----------------------------------- |
| GRU  | update, reset, new          | `[0, 1, 2]`     | `Wb` and `Rb` on separate `Linear`s |
| LSTM | input, forget, output, cell | `[0, 2, 1, 3]`  | `Wb + Rb` folded, other zeroed      |
| RNN  | gate                        | `[0]`           | `Wb + Rb` folded, other zeroed      |

Bidirectional falls out for free: `BiGru`, `BiLstm` and `BiRnn` are `{ forward, reverse }`, which is
the same `forward.`/`reverse.` prefix pair the snapshot paths already use, so the emitter just
prefixes the field access.

On the onnx-ir side, `lift_constants` for these three ops is now all-or-nothing across `W`, `R` and
`B`. Lifting them independently could leave a model where one weight is a lifted `Static` (name
cleared) and another is a graph input: the runtime path would then have an input it cannot name.
Refusing to lift any of them when one is dynamic keeps the unlifted constants as named `Constant`
node outputs, which the runtime path can reference like any other value. The rule is
`processor::lift_all_or_none(node, indices)`, general rather than RNN-specific, because
`batch_norm.rs` already open-codes the same reasoning and the next weighted op will want it too.

Two things surfaced that were not in the issue:

- **`layout=1` put the direction axis on the wrong dimension.** ONNX's batch-first layout moves
  `num_directions` in the *outputs* too: `Y` becomes `[batch, seq, dirs, hidden]` and `Y_h` becomes
  `[batch, dirs, hidden]`. LSTM and RNN unsqueezed at a fixed dim 1 and 0 regardless of layout, and
  GRU had `Y` right and `Y_h` wrong. Bidirectional plus `layout=1` needs a `swap_dims(0, 1)` on the
  final state, which nothing did. This was unreachable before: the only upstream tests with
  `layout=1` are the three `*_batchwise` rows, and every one of them was already failing for the
  weights reason. Fixing the weights is what made the shape bug observable - the first pass promoted
  8 of 11 rows, and the 3 that held out were exactly the batchwise ones.
- **The snapshot tests were recording the bug.** `Argument::new` defaults to `ValueSource::Dynamic`,
  so every GRU/LSTM/RNN codegen test built a node whose weights are, semantically, graph inputs. The
  accepted snapshots showed `pub fn forward(&self, input: Tensor<3>, W: Tensor<3>, R: Tensor<3>, B:
  Tensor<2>)` with a body that reads `self.gru1` and never touches `W`/`R`/`B` - the defect, frozen
  as an expectation. The static-path tests now mark their weights as lifted initializers explicitly,
  and there are new snapshot tests for the runtime path.

Scoreboard: 11 rows promoted from `fail-compare` to `pass` (4 GRU, 3 LSTM, 4 RNN). Harness tests went
from 819 to 830, all green. Four integration tests were added under `crates/onnx-tests/tests/`: one
per op going through `from_file`, because the official harness constructs with `Model::new` while
`#458`'s headline symptom is `from_file` panicking, and those compare against `ReferenceEvaluator`.

The fourth is a bidirectional GRU, which nothing upstream covers and which `ReferenceEvaluator`
cannot serve either - its GRU raises `NotImplementedError` for `num_directions=2`. It uses the same
weights as initializers as its own oracle and compares the two models element-wise, which pins the
one piece of the layout still written twice: flipping GRU's `BiasLayout` to `Merged` fails it, while
a wrong `GateLayout` does not, because that const drives both paths and moves them together.

Unchanged and still rejected with a clear message: LSTM peephole connections (input `P`) and
`sequence_lens` on all three.

Two latent bugs were left alone as out of scope, both predating this work and both affecting the
static path equally:

- `collect_*_snapshots` slices a direction with `.squeeze::<2>()`, which drops *every* size-1
  dimension, so a GRU with `input_size == 1` or `hidden_size == 1` would squeeze the wrong axis and
  fail the rank check. The generated runtime path uses `select_dim`/`slice_dim` and is not exposed.
- Nothing validates a declared `W`/`R` shape against the `hidden_size` and `direction` attributes.
  An undersized weight panics in Burn's slice check, but an oversized one is silently truncated -
  a `[2, ...]` W in a `direction="forward"` model quietly uses direction 0 only. This belongs in
  `infer_types` as a `ProcessError` where the static shape is known.

## Tier 3

- **#50 / #51 metal backend.** Dozens of ops fail and YOLO11x diverges by 295 max-abs. Correctness
  on the backend people ship on outweighs op-count wins. Root cause probably belongs upstream in
  burn, but it surfaces here.
- **Resize shares every gap Upsample just closed (surfaced by the item 1 review).** None of it is
  new and none of it blocked item 1, but it is all in `crates/burn-onnx/src/import/burn/node/resize.rs`
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
  this shape (`.expect("Config extraction failed")`); Upsample is just the first to have checks
  that can realistically fire there. Reproduced with a `Constant -> Identity -> Upsample` graph
  carrying scales of 1.75.
- **The static-vs-runtime weight decision is stated on both sides of the crate boundary (surfaced by
  the item 6 review).** `lift_all_or_none`'s liftability test in onnx-ir and `weights_are_runtime` in
  burn-onnx encode the same invariant with two non-complementary predicates over a four-variant
  `ValueSource`, kept in sync by hand. `BatchNormalizationNode` already shows the deeper answer:
  decide once in `extract_config` and carry it in the IR config (`BatchNormConfig::Static |
  Runtime`), so codegen matches rather than re-derives. Doing the same for GRU/LSTM/RNN churns every
  positional `Config::new(...)` in their tests, which is why it is not in item 6.
- **Generated `use` lines are predicted per node.** `BurnImports` emits bare `use` with no
  `#[allow(unused_imports)]`, so LSTM and RNN carry a `needs_module_type` flag purely to avoid
  importing a type the runtime path never names. GRU needs none of it because it uses fully-qualified
  `burn::nn::gru::Gru` paths. Either adopt GRU's style in the other two, or allow unused imports in
  the generated block once.
- **#280 shape propagation through Where/Mul/ConstantOfShape.** Blocks RF-DETR without an `onnxsim`
  pre-pass.
- **#371 Kokoro residual 1.3x.** Established as f32 drift through HiFi-GAN resblocks, not fixable
  here. Close or move to burn.

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

Items 1, 2 and 6 are done, and #460 landed alongside item 2. That clears the known
silent-wrong-answer bugs from the board; what is left is test-count work against an honest baseline.

Item 5's top three rows (34 + 22 + 22 = 78 of the 103 remaining `skip-compile` rows) are probably
two fixes, which makes that bucket the largest remaining win. Item 3 is the next-largest and the
better-understood one: Conv/ConvTranspose/LayerNorm all have a functional entry point in
`burn-tensor` to route through, which is what item 6 turned out not to have. Then 4.

Item 6 did not produce a helper item 3 can reuse — the two share a symptom, not a mechanism — so
item 3 starts from the `runtime_scalar_to_native` extraction proposed in the #314 thread, not from
`node/rnn_common.rs`.
