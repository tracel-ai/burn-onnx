# ONNX Tests

End-to-end tests for ONNX to Burn conversion. These tests ensure ONNX models are accurately
converted into Burn source code that compiles without errors and produces the same outputs as the
original ONNX model.

## Quick Start

```sh
# Set up Python environment
cd crates/onnx-tests
uv sync

# Run tests
cargo test
```

## Directory Structure

- `tests/<op_name>/`: Each operator has its own directory
- `tests/<op_name>/<op_name>.py`: Python script that generates the ONNX model
- `tests/<op_name>/<op_name>.onnx`: Generated ONNX model
- `tests/<op_name>/mod.rs`: Test implementation
- `tests/simplify/`: Simplification comparison tests (see below)

## Simplification Testing

The ONNX-IR pipeline supports an optional simplification pass (`ModelGen::simplify(true)`) that folds
shape computations into constants at codegen time. The `tests/simplify/` directory contains
purpose-built ONNX models that exercise specific simplification patterns.

**How it works:**

- The `build.rs` generates three sets of models:
  1. `model/` - Main models with `.simplify(false)` (used by existing operator tests)
  2. `model_simplified/` - Simplify test models with `.simplify(true)`
  3. `model_unsimplified/` - Same simplify test models with `.simplify(false)`
- Each test in `tests/simplify/mod.rs` runs both the simplified and unsimplified version with
  identical inputs and asserts the outputs match
- The `include_simplified_models!` macro (in `test_mod.rs`) includes models from both directories

**Adding a new simplification pattern test:**

1. Add a new model-generating function in `tests/simplify/gen_models.py`
2. Run the script: `uv run --script tests/simplify/gen_models.py`
3. Add the new `.onnx` file to `add_simplify_inputs()` in `build.rs`
4. Add the model name to `include_simplified_models!` in `tests/simplify/mod.rs`
5. Write a comparison test in `tests/simplify/mod.rs`

**Existing pattern categories:** shape folding, gather-on-shape, slice-on-shape, concat-shapes,
reshape-from-shape, binary-ops-on-shape, cast-shape, where-on-shapes, expand-from-shape,
constant-of-shape.

## Running Tests

```sh
# Default backend (NdArray)
cargo test

# WGPU backend
cargo test --features test-wgpu

# LibTorch backend
cargo test --features test-tch

# Specific test
cargo test --test test_mod softmax::test_softmax
```

## Resources

- [Development Guide](../../DEVELOPMENT-GUIDE.md#integration-testing) - Detailed guide for creating
  tests
- [Supported ONNX Operators](../../SUPPORTED-ONNX-OPS.md) - List of supported operators
