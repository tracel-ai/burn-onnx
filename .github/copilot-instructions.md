# Copilot Instructions for burn-onnx

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

examples/
├── onnx-inference/          # Standard inference example
├── image-classification-web/ # WASM/WebGPU example
└── raspberry-pi-pico/       # no_std embedded example (excluded)
```

## Crate Responsibilities

### onnx-ir
- Parses ONNX protobuf files into a clean Intermediate Representation
- 5-phase pipeline: Initialization → Node Conversion → Type Inference → Post-processing → Finalization
- Each ONNX operator has a `NodeProcessor` implementation
- Node structs contain: `name`, `inputs`, `outputs`, `config`

### burn-onnx
- Converts onnx-ir nodes to Burn Rust code
- Implements `NodeCodegen` trait for each node type
- Generates `.rs` files and `.burnpack` weight files
- Entry point: `ModelGen` builder

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

### burn-onnx Patterns
- Implement `NodeCodegen<PS>` directly on onnx-ir node types
- Use `scope.arg()` for automatic tensor/scalar/shape handling
- Use `quote!` macro for code generation
- Add snapshot tests with `insta` crate

### Testing
- Unit tests go in the same file as implementation
- Integration tests in `crates/onnx-tests/tests/<op_name>/`
- Python scripts generate ONNX models for testing
- Use `torch.manual_seed(42)` for reproducibility

## Code Review Guidelines

### What to Check

1. **Type Safety**
   - Ensure proper error handling (no unwrap in library code except tests)
   - Check tensor rank and dtype consistency
   - Verify configuration extraction handles all ONNX attributes

2. **Code Generation**
   - Generated code should compile without warnings
   - Check for proper variable naming (no conflicts)
   - Verify correct Burn API usage

3. **Testing**
   - New operators need both unit tests and integration tests
   - Edge cases: empty tensors, single elements, broadcasting
   - Multiple input shapes and data types

4. **Documentation**
   - Public items need doc comments
   - Update SUPPORTED-ONNX-OPS.md for new operators
   - Include examples in doc comments for complex APIs

### Common Issues

- Missing `#[cfg(feature = "...")]` guards
- Incorrect tensor dimension handling
- Not handling optional ONNX inputs
- Forgetting to register new nodes in dispatch macro
- Using `panic!` instead of returning `Result`

## Dependencies

- `burn` crates are from git (tracel-ai/burn)
- `protobuf` for ONNX parsing
- `quote`/`proc-macro2`/`syn` for code generation
- `insta` for snapshot testing

## Feature Flags

- `mmap` - Memory-mapped file loading (default enabled in onnx-ir)
- `std` - Standard library support (required for most functionality)

## Links

- [Development Guide](../DEVELOPMENT-GUIDE.md) - Detailed implementation guide
- [Supported Operators](../SUPPORTED-ONNX-OPS.md) - Operator support table
- [Burn Documentation](https://burn.dev/book/) - Main Burn framework docs
