# Claude Instructions for burn-onnx

This repository contains the ONNX import functionality for the Burn deep learning framework.

## Project Structure

```
crates/
├── burn-onnx/       # Main crate - converts ONNX IR to Burn code
├── onnx-ir/         # ONNX Intermediate Representation parser
├── onnx-ir-derive/  # Derive macros for onnx-ir
├── onnx-tests/      # End-to-end integration tests
├── burn-import/     # Legacy crate (deprecated, re-exports burn-onnx)
└── model-checks/    # Real-world model validation (excluded from workspace)

onnx-spec/
├── fetch-specs.py   # Script to fetch/update ONNX operator specs
└── ops/             # Per-operator markdown specs (auto-generated)

examples/
├── onnx-inference/          # Standard inference example
├── image-classification-web/ # WASM/WebGPU example
└── raspberry-pi-pico/       # no_std embedded example (excluded)
```

## Crate Responsibilities

### onnx-ir

- Parses ONNX protobuf files into a clean Intermediate Representation
- 5-phase pipeline: Initialization → Node Conversion → Type Inference → Post-processing →
  Finalization
- Each ONNX operator has a `NodeProcessor` implementation
- Node structs contain: `name`, `inputs`, `outputs`, `config`
- **Important**: Should extract and preserve ALL ONNX attributes faithfully, even if Burn doesn't
  support them yet. This allows onnx-ir to be reused by other projects
- **Important**: Config structs should mirror ONNX semantics, not Burn semantics. Do NOT pre-compute
  or translate ONNX attributes into Burn-specific values (e.g., do not resolve `auto_pad` into
  explicit padding values). Store the original ONNX attribute and let burn-onnx handle the
  translation during code generation

### burn-onnx

- Converts onnx-ir nodes to Burn Rust code
- Implements `NodeCodegen` trait for each node type
- Generates `.rs` files and `.burnpack` weight files
- Entry point: `ModelGen` builder
- **Important**: Rejection of unsupported features happens HERE, not in onnx-ir. If Burn API doesn't
  support a configuration, burn-onnx should emit a clear error during code generation
- **Important**: Translation from ONNX semantics to Burn semantics happens HERE. For example,
  resolving ONNX `auto_pad` into explicit padding values for Burn's API is a burn-onnx
  responsibility, not onnx-ir's
- **Important**: Always generate the simplest and most efficient Burn Rust code possible. Avoid
  emitting dead code, no-op loops, or redundant operations when the result can be determined at
  codegen time
- **Important**: When in doubt about Burn APIs, search online for the latest documentation rather
  than guessing

## Coding Conventions

### Rust Style

- Edition 2024
- Use `#[derive(Debug, Clone)]` on public types
- Prefer `thiserror` for error types
- Use `log` crate for logging (not `println!`)
- Document public APIs with `///` doc comments

### ONNX-IR Patterns

- Node processors are `pub(crate)` - only the node structs and configs are public
- Use `NodeBuilder` derive macro for test builders
- Configuration structs should derive `Debug, Clone, Default` when possible
- Type inference happens in processors, not in codegen
- **Always produce partial `static_shape`**: `infer_types()` should always set `static_shape` to
  `Some(vec![...])` on tensor outputs, using `None` for unknown dimensions and `Some(value)` for
  known ones. Never leave `static_shape` as `None` when the output rank is known. Start with
  `tensor.static_shape.clone().unwrap_or_else(|| vec![None; rank])` and fill in whatever dimensions
  can be determined from inputs, weights, or config. This enables per-dimension merging via
  `merge_static_shape()`
- **Strive for full ONNX opset coverage** - extract all attributes even if not yet used by burn-onnx
- **Support all opsets** - when implementing operators, set `min_opset` to the earliest opset version
  that introduced the operator (not the latest version). Each `onnx-spec/ops/<OpName>.md` file shows
  the first introduced opset and full version history
- Config structs should include all ONNX operator attributes, using `Option<T>` for optional ones
- **Reference `onnx-spec/ops/<OpName>.md`** for the official spec when implementing or reviewing
  operators (attributes, inputs/outputs, type constraints)
- **Declarative node architecture**: General processing in the onnx-ir framework (pipeline phases,
  graph state, type inference loop, etc.) must NOT contain node-type-specific logic. All
  node-specific behavior is declared in `NodeProcessor` implementations. If a general module needs
  to handle a particular node type differently, that logic belongs in the node's processor, not in
  the framework code
- **Optional input handling**: ONNX uses empty string `""` for "optional input not provided". Use
  `node.get_input(index)` which returns `None` for out-of-bounds or optional inputs. In
  `lift_constants()` (which needs `&mut`), guard with `!node.inputs[N].is_optional()`. Never check
  `name.is_empty()` to detect optional inputs; use `is_optional()` instead
- **Processor registration is mandatory**: Every node type must be registered in `registry.rs`.
  Unregistered types fall through to `DefaultProcessor` (which does `same_as_input`), producing
  wrong type info for ops that change tensor rank. Type inference pre-checks for unregistered ops
  and reports them all before processing
- **`ProcessError` has a `Display` impl**: Use it for user-facing messages. Avoid formatting with
  `{:?}` which exposes variant names like `Custom("...")`

### burn-onnx Patterns

- Implement `NodeCodegen<PS>` directly on onnx-ir node types
- Use `scope.arg()` for automatic tensor/scalar/shape handling. Note: `scope.arg()` may add
  `.clone()` when the tensor is used by multiple downstream nodes. If you also need the tensor twice
  within the same `forward()`, using `#input.clone()` in `quote!` may produce a double-clone in that
  case, which is acceptable
- **Scope temporary variables in generated code**: When a node's codegen introduces temporary
  variables (e.g., `indices`, intermediate tensors), wrap them in a block expression to avoid name
  collisions with other nodes: `let #output = { let tmp = ...; tmp.op() };`
- Use `quote!` macro for code generation
- Add `insta` snapshot tests for ALL code generation branches - each config option, each input type
  variant, optional vs required inputs should have test coverage
- **Inline snapshots only** - use `assert_snapshot!(code, @r"...")` with embedded expected output,
  not external `.snap` files

### Testing

- Unit tests go in the same file as implementation
- Integration tests in `crates/onnx-tests/tests/<op_name>/`
- Simplification comparison tests in `crates/onnx-tests/tests/simplify/`
- Use `torch.manual_seed(42)` or `np.random.seed(42)` for reproducibility

### Bug Fixes

- Every bug fix **must** include an integration test that reproduces the bug
- Workflow: write a failing test first, then fix the code to make it pass
- The test should fail without the fix and pass with it

### Simplification

- `ModelGen::simplify(true)` enables an optional ONNX-IR pass that folds shape computations into
  constants at codegen time (e.g., `Shape(x)` with static dims becomes a constant array)
- Existing operator tests in `crates/onnx-tests/` use `.simplify(false)` to test unsimplified
  codegen
- Dedicated tests in `crates/onnx-tests/tests/simplify/` have their own purpose-built ONNX models
  that are compiled both with and without simplification to verify outputs match
- The `build.rs` generates three model sets: `model/` (main, unsimplified), `model_simplified/`, and
  `model_unsimplified/` (the latter two for simplify comparison tests only)
- When adding a new simplification pattern, add a test model via `tests/simplify/gen_models.py` and
  a comparison test in `tests/simplify/mod.rs`

### Python Test Scripts

Use `uv` inline script format for self-contained test scripts:

```python
#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "torch==2.1.1",
# ]
# ///

import torch
import onnx
from onnx.reference import ReferenceEvaluator
```

Always use `onnx.reference.ReferenceEvaluator` to compute expected outputs - this is the official
ONNX reference implementation and serves as ground truth.

For newer ONNX ops (opset 21+), `torch.onnx.export` often can't emit the op directly. Use
`onnx.helper.make_node` / `onnx.helper.make_model` to construct the ONNX model manually instead.

## Adding a New ONNX Operator

1. **onnx-ir**: Create node processor in `crates/onnx-ir/src/node/<op>.rs`
   - Read `onnx-spec/ops/<OpName>.md` for the full operator spec
   - Define config struct with ALL ONNX attributes
   - Implement `NodeProcessor` trait
   - Register in these files:
     - `crates/onnx-ir/src/node/mod.rs` - add `pub mod <op>;`
     - `crates/onnx-ir/src/ir/node.rs` - move from `unsupported::<Op>Node` to `<op>::<Op>Node`
     - `crates/onnx-ir/src/registry.rs` - register the processor

2. **burn-onnx**: Create codegen in `crates/burn-onnx/src/burn/node/<op>.rs`
   - Implement `NodeCodegen<PS>` for the onnx-ir node type
   - Add `insta` snapshot tests for all code generation branches
   - Register in these files:
     - `crates/burn-onnx/src/burn/node/mod.rs` - add `pub(crate) mod <op>;`
     - `crates/burn-onnx/src/burn/node_codegen.rs` - add `<Op>` to dispatch macro

3. **onnx-tests**: Create integration test in `crates/onnx-tests/tests/<op>/`
   - Python script to generate ONNX model (use uv script format)
   - Use `ReferenceEvaluator` for expected outputs
   - Rust test in `mod.rs`
   - Register in these files:
     - `crates/onnx-tests/build.rs` - add `.input("tests/<op>/<op>.onnx")`
     - `crates/onnx-tests/tests/test_mod.rs` - add `pub mod <op>;`

4. **Documentation**: Update `SUPPORTED-ONNX-OPS.md`

## Common Commands

```sh
# Run all tests
cargo test

# Run specific crate tests
cargo test -p onnx-ir
cargo test -p burn-onnx
cargo test -p onnx-tests

# Run validation (formatting, linting, tests)
cargo xtask validate

# Generate code from ONNX model
cargo run -p burn-onnx -- model.onnx ./out

# Update insta snapshots
cargo insta review
```

## Key Files to Know

- `crates/onnx-ir/src/processor.rs` - NodeProcessor trait definition, ProcessError, DefaultProcessor
- `crates/onnx-ir/src/registry.rs` - Processor registration (unregistered types use DefaultProcessor)
- `crates/onnx-ir/src/phases/type_inference.rs` - Iterative type inference with unsupported op detection
- `crates/onnx-ir/src/ir/node.rs` - Node enum and define_node_enum! macro
- `crates/burn-onnx/src/burn/node_codegen.rs` - Codegen dispatch macro
- `crates/burn-onnx/src/burn/graph.rs` - Graph code generation
- `SUPPORTED-ONNX-OPS.md` - Operator support table
- `DEVELOPMENT-GUIDE.md` - Detailed implementation guide
- `onnx-spec/ops/<OpName>.md` - Official ONNX operator specs (update with
  `./onnx-spec/fetch-specs.py`)

## Dependencies

- `burn` crates are from git (tracel-ai/burn)
- `protobuf` for ONNX parsing
- `quote`/`proc-macro2`/`syn` for code generation
- `insta` for snapshot testing

## Feature Flags

- `mmap` - Memory-mapped file loading (default enabled in onnx-ir)
- `std` - Standard library support (required for most functionality)
