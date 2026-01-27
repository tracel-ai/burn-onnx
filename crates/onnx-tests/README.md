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
