# onnx-official-tests

Native regression gate against the upstream
[ONNX backend test suite](https://github.com/onnx/onnx/tree/main/onnx/backend/test).

Runs the full set of ~1600 upstream node tests against `burn-onnx`- generated Rust code on every
`cargo test`, with no Docker, no Python, and no network. Each test is declared in
`expectations.toml` with a status describing how it's expected to behave today; the build script
feeds the pass-list to `burn-onnx::ModelGen` and emits one `#[test]` function per test whose
generated `Model` has a harness-compatible signature.

```sh
cargo test -p onnx-official-tests
```

## What's in here

```text
crates/onnx-official-tests/
├── Cargo.toml
├── build.rs                # data-driven codegen + per-test harness emission
├── expectations.toml       # declarative source of truth for every upstream test
├── vendor/node/            # upstream test data, vendored from onnx v1.19.0
│   └── test_<name>/
│       ├── model.onnx
│       └── test_data_set_0/{input_*.pb, output_0.pb}
├── src/
│   ├── lib.rs
│   ├── expectations.rs     # parse expectations.toml
│   └── pb_loader.rs        # decode TensorProto .pb files (multi-dtype)
└── tests/
    ├── backend.rs          # selects burn backend by feature flag
    └── test_mod.rs         # includes build.rs-generated harness + drift check
```

`build.rs` emits three files into `$OUT_DIR` which `tests/test_mod.rs` `include!`s:

- `models.rs` — one `pub mod <name>` declaration per generated model, exposing
  `generated::test_<name>::Model`.
- `harness.rs` — one `#[test] fn <name>()` per test whose input/output signatures the harness can
  express statically. Each test body is a straight-line sequence of `load_tensor` /
  `Tensor::from_data` / `model.forward` / `assert_approx_eq` calls with ranks and dtypes hardcoded
  by the build script.
- `manifest.rs` — the lists `HARNESS_TESTS` and `CODEGEN_ONLY_TESTS` used by the drift-check test to
  verify three-way agreement with `expectations.toml` and `vendor/node/`.

## Vendored test data

The data under `vendor/node/` is copied verbatim from [`onnx/onnx`](https://github.com/onnx/onnx) at
tag `v1.19.0`. It's a direct commit (~35 MB across ~1600 test directories) rather than a submodule
or LFS-backed tree so clean checkouts stay self-contained and CI needs no network access.

To refresh against a new ONNX release, use the xtask helper:

```sh
cargo xtask refresh-onnx-tests --version 1.19.0
# sanity-check the URL and paths first:
cargo xtask refresh-onnx-tests --version 1.19.0 --dry-run
```

The helper downloads the release tarball, extracts `onnx/backend/test/data/node/`, and replaces
`vendor/node/` with its contents. After refreshing, run `cargo test -p onnx-official-tests`; the
drift check will flag any new or removed test directories, and any tests whose actual status has
changed will appear as failures ready to be demoted or promoted in `expectations.toml`.

## Expectations

`expectations.toml` is the declarative source of truth. Every vendored test has exactly one entry,
with a status that describes how `burn-onnx` is expected to behave on that model today:

| status         | meaning                                                            | handled by `build.rs`        |
| -------------- | ------------------------------------------------------------------ | ---------------------------- |
| `pass`         | codegen + compile + output match the reference                     | fed to `ModelGen`, harnessed |
| `skip-codegen` | `onnx2burn` panics or refuses the model (unsupported op or config) | not fed to `ModelGen`        |
| `skip-compile` | codegen succeeds but the generated Rust does not compile           | not fed to `ModelGen`        |
| `fail-compare` | compiles and runs but produces incorrect output                    | not fed to `ModelGen`        |
| `flaky`        | intermittent, ignored by the gate                                  | not fed to `ModelGen`        |

Only `pass` entries are compiled, harnessed, and gated. Everything else is documentation that
describes the current gap — a contributor fixing a skip-codegen case can promote it to `pass`, and
the next `cargo test` either confirms the fix or reveals a more-specific failure (e.g. a fresh
`fail-compare`) that gets demoted.

Optional fields per entry: `reason`, `tracking` (issue/PR ref), `wontfix` (bool; `true` means out of
scope rather than an intentional gap to close).

The initial M2 classification was sourced from tracel-ai/burn-onnx#314 (skip-codegen list from the
~656 codegen failures) and tracel-ai/burn-onnx#311 (fail-compare list from the op- family comparison
failures). The first full `cargo test` run after populating `expectations.toml` from those sources
produced a set of classification errors — tests marked `pass` that actually fail codegen, compile,
or comparison — which were iteratively demoted until the suite went green.

Not every `pass` entry turns into a `#[test]` function. `build.rs` skips harness emission (but still
runs codegen, so the generated `Model` compiles) for a few edge cases:

- **Rank-0 (scalar) I/O.** `burn-onnx` lowers scalars to `ScalarNative` or `ScalarTensor` rather
  than `Tensor<B, 0>`; the harness doesn't yet cover that branch.
- **Exotic dtypes** in inputs or outputs (FLOAT16, BFLOAT16, FLOAT8*, INT4, UINT4, FLOAT4E2M1,
  STRING, COMPLEX*). `pb_loader` decodes only the dtype set burn supports natively.
- **Non-tensor forward return types** such as `[i64; N]` that `burn-onnx` emits for ONNX
  `Shape`/`Size`/etc. ops whose outputs are known at codegen time. The build script post-processes
  each generated `.rs` file to detect this and downgrades the test to codegen-only.

Tests in any of the above categories appear in `CODEGEN_ONLY_TESTS` in the generated manifest
instead of `HARNESS_TESTS`. The drift check asserts their union equals the pass-set so a future
classification mistake can't silently drop a test from both.

## What this crate is not

- Not a replacement for `crates/onnx-tests/` — those hand-crafted tests exercise targeted
  regressions and stay the canonical place for "did this specific bug come back?".
- Not a replacement for `scoreboard/` — that submits to the upstream ONNX Backend Scoreboard via
  Docker. This crate is the local CI gate.

## Roadmap

Tracked in [#315](https://github.com/tracel-ai/burn-onnx/issues/315).

- **M1** — scaffold: small known-passing test set, drift check, declarative expectations. Done
  (#318).
- **M2** — full coverage: vendor ~1600 upstream node tests, populate expectations from #314/#311,
  data-driven harness (no per-test macros), multi-dtype `pb_loader`,
  `cargo xtask refresh-onnx-tests`. **This PR.**
- **M3** — CI sharding: matrix test job keyed on op-name prefix so wall time is O(max per-shard)
  instead of O(total).
- **M4** — PR-comment delta: `cargo xtask diff-expectations --base=origin/main` that posts a summary
  of promotions, regressions, and still-failing counts.
- **M5** — `--update-expectations` convenience flag: "accept the new greens" workflow for local
  development.
