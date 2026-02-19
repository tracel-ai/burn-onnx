# AGENTS.md - burn-onnx

ONNX import for the Burn deep learning framework. Converts ONNX models to Rust
source code and `.bpk` weight files.

For detailed architecture, pipeline phases, code examples, and step-by-step operator implementation
walkthrough, read `DEVELOPMENT-GUIDE.md`.

## Project Structure

```
crates/
├── burn-onnx/       # Converts ONNX IR to Burn Rust code (codegen)
├── onnx-ir/         # ONNX Intermediate Representation parser
├── onnx-ir-derive/  # Derive macros for onnx-ir
├── onnx-tests/      # End-to-end integration tests
├── burn-import/     # Legacy crate (deprecated, re-exports burn-onnx)
└── model-checks/    # Real-world model validation (excluded from workspace)

onnx-spec/ops/       # Per-operator markdown specs (reference material)
```

## Architecture Rules

**onnx-ir** and **burn-onnx** have a strict separation of concerns:

| Responsibility | Where |
|---|---|
| Parse ONNX protobuf into IR | onnx-ir |
| Extract ALL ONNX attributes faithfully | onnx-ir |
| Type inference, static shape inference | onnx-ir (`NodeProcessor`) |
| Structural validation (e.g. "got 3D, need 2D") | onnx-ir (`ProcessError`) |
| Translate ONNX semantics to Burn semantics | burn-onnx |
| Reject unsupported Burn features | burn-onnx (clear error, not panic) |
| Generate Rust code | burn-onnx (`NodeCodegen`) |

Key principles:
- **onnx-ir mirrors ONNX, not Burn.** Config structs store original ONNX attributes. Do not
  pre-compute Burn-specific values (e.g., don't resolve `auto_pad` into padding values)
- **No panics in codegen.** Use `ProcessError` in onnx-ir for validation. Panics in burn-onnx
  crash the build with poor error messages
- **Prefer Burn tensor APIs over manual loops.** Check the Burn API before generating element-wise
  code. Native tensor ops are orders of magnitude faster. When in doubt about Burn APIs, search
  online rather than guessing
- **Declarative node architecture.** Framework code must not contain node-type-specific logic. All
  node-specific behavior lives in `NodeProcessor` implementations
- **Don't reject unknown ONNX attributes.** Extract what you need, ignore the rest. Future opsets
  may add new attributes

## Coding Conventions

### Rust

- Edition 2024
- `#[derive(Debug, Clone)]` on public types
- `thiserror` for errors, `log` for logging (not `println!`)
- `///` doc comments on public APIs

### onnx-ir

- Node processors are `pub(crate)`; only node structs and configs are public
- Config structs: `#[derive(Debug, Clone, Default)]`, include ALL ONNX attributes (`Option<T>` for
  optional ones)
- Set `min_opset` to the earliest opset that introduced the operator (check
  `onnx-spec/ops/<OpName>.md`)
- Every processor must be registered in `registry.rs`
- Use `node.get_input(index)` for optional inputs (returns `None` for absent/optional). Never check
  `name.is_empty()`; use `is_optional()` instead
- `ProcessError` has a `Display` impl. Format with `{}`, not `{:?}`
- See `DEVELOPMENT-GUIDE.md` for `static_shape`, constant lifting, and `NodeProcessor` trait details

### burn-onnx

- Implement `NodeCodegen` directly on onnx-ir node types
- Use `scope.arg()` for inputs: handles clone tracking for on-device values (`Tensor`,
  `ScalarTensor`) and bare idents for host values (`ScalarNative`, `Shape`)
- Use `arg_to_ident()` only for outputs and host-side values. Never use it for `ScalarTensor`
  inputs (it skips clone tracking)
- Scope temporary variables in block expressions to avoid name collisions
- `insta` snapshot tests for ALL codegen branches (inline snapshots only:
  `assert_snapshot!(code, @r"...")`)

### Testing

- Unit tests in the same file as implementation
- Integration tests in `crates/onnx-tests/tests/<op_name>/`
- Bug fixes **must** include an integration test (write failing test first, then fix)
- Use `torch.manual_seed(42)` / `np.random.seed(42)` for reproducibility
- Cover at least one non-default configuration per operator
- Python test scripts use `uv` inline script format with `onnx.reference.ReferenceEvaluator` as
  ground truth

## Adding a New Operator

Read `DEVELOPMENT-GUIDE.md` for the full walkthrough with code examples. Checklist:

1. **onnx-ir**: `crates/onnx-ir/src/node/<op>.rs`
   - Read `onnx-spec/ops/<OpName>.md` for the full spec
   - Define config struct, implement `NodeProcessor`
   - Register in: `node/mod.rs`, `ir/node.rs` (macro), `registry.rs`

2. **burn-onnx**: `crates/burn-onnx/src/burn/node/<op>.rs`
   - Implement `NodeCodegen`, add `insta` snapshot tests
   - Register in: `node/mod.rs`, `node_codegen.rs` (dispatch macro)

3. **onnx-tests**: `crates/onnx-tests/tests/<op>/`
   - Python script (uv format) + Rust test
   - Register in: `build.rs`, `test_mod.rs`

4. Update `SUPPORTED-ONNX-OPS.md`

## Code Review Checklist

- Config structs include ALL ONNX attributes (don't skip because burn-onnx doesn't use them yet)
- No `unwrap` in library code (tests are fine)
- No `panic!` for structural validation (use `ProcessError`)
- Generated code compiles without warnings
- Snapshot tests cover each config variant and input type
- New operators have both unit and integration tests
- Processor is registered in `registry.rs` and dispatch macro

## Common Commands

```sh
cargo test                          # Run all tests
cargo test -p onnx-ir               # Test specific crate
cargo test -p burn-onnx
cargo test -p onnx-tests
cargo xtask validate                # Format, lint, test
cargo run -p burn-onnx --bin onnx2burn -- model.onnx ./out  # Generate code from ONNX
cargo insta review                  # Review snapshot changes
```

## Key Files

- `crates/onnx-ir/src/processor.rs` - `NodeProcessor` trait, `ProcessError`, `DefaultProcessor`
- `crates/onnx-ir/src/registry.rs` - Processor registration
- `crates/onnx-ir/src/ir/node.rs` - `Node` enum and `define_node_enum!` macro
- `crates/burn-onnx/src/burn/node_codegen.rs` - Codegen dispatch macro
- `crates/burn-onnx/src/burn/graph.rs` - Graph code generation
- `DEVELOPMENT-GUIDE.md` - Full implementation guide with code examples
- `SUPPORTED-ONNX-OPS.md` - Operator support table
- `onnx-spec/ops/<OpName>.md` - Official ONNX operator specs
