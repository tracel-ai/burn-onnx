# ONNX-IR

Part of the [burn-onnx](https://github.com/tracel-ai/burn-onnx) project.

ONNX-IR is a pure Rust library for parsing ONNX models into an intermediate representation (IR) that
can be used to generate code for various ML/DL frameworks. It's a core component of the Burn model
import system, providing a clean abstraction layer between ONNX protobuf structures and Burn's
tensor operations.

## Overview

ONNX-IR converts ONNX protobuf models into a clean intermediate representation through a 5-phase
pipeline. The resulting IR provides:

- **Enum-based node representation**: Each node is a variant of the `Node` enum with
  operation-specific configuration
- **Typed inputs/outputs**: All node arguments are validated with type information
- **Pre-extracted configuration**: Attributes are parsed into strongly-typed config structs
- **Static tensor data**: Constant values are available for constant folding
- **Support for 100+ ONNX operators**: Including control flow (`If`, `Loop`, `Scan`)
- **Tolerant of unknown operators**: Operators from custom or vendor domains parse as
  `Node::Custom` rather than failing, so any model can be inspected

For detailed architecture information, see the
[Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md).

## Usage

ONNX-IR is typically used through the `burn-onnx` crate, but can also be used standalone:

```rust
use onnx_ir::{OnnxGraphBuilder, OnnxGraph, Node};

// Parse an ONNX model from file (uses mmap when available)
let graph: OnnxGraph = OnnxGraphBuilder::new()
    .parse_file("path/to/model.onnx")?;

// Or parse from bytes
let graph = OnnxGraphBuilder::new().parse_bytes(&model_bytes)?;

// Work with the IR - nodes are represented as an enum
for node in &graph.nodes {
    println!("Node: {}", node.name());

    // Pattern match on node type to access operation-specific configuration
    match node {
        Node::Softmax(softmax_node) => {
            println!("  Softmax on axis {}", softmax_node.config.axis);
        }
        Node::Conv2d(conv_node) => {
            println!("  Conv2d with kernel size {:?}", conv_node.config.kernel_size);
        }
        // Operators this crate does not implement keep their ONNX identity
        Node::Custom(custom) => {
            println!("  {}::{} (opset {})", custom.domain, custom.op_type, custom.opset);
        }
        _ => {}
    }
}
```

## Unknown and Custom Operators

An operator is resolved by its full ONNX identity `(domain, op_type)`. Op types in the standard
domains (`""`, `ai.onnx`, `ai.onnx.ml`) map to built-in `Node` variants; anything else — a vendor
domain like `com.microsoft`, or an op type this crate does not implement — becomes
`Node::Custom`, carrying its raw op type, domain, per-domain opset, attributes, and inputs
(including readable constant data). Parsing does not fail, so unknown models stay inspectable.

Type inference for those nodes falls back to a best-effort guess. To supply real inference,
implement `CustomOpInference` and register it:

```rust
let graph = OnnxGraphBuilder::new()
    .with_custom_op_inference(Arc::new(my_hooks))
    .parse_file("path/to/model.onnx")?;
```

With hooks registered, a pre-pass reports every custom operator they do not cover in one error
(`Error::MissingCustomOpHooks`) before type inference runs, rather than failing later on a
cascading type error.

Code generation for custom operators lives in `burn-onnx`, whose `CustomOp` trait pairs inference
with emitted Rust; see its
[Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md).

## Memory-Mapped Loading

By default, ONNX-IR uses memory-mapped file I/O (mmap) when loading models from files. This
provides:

- **Reduced memory usage**: Tensor data is read directly from the file on demand
- **Faster startup**: No need to copy the entire file into memory upfront
- **Lazy loading**: Data is only copied when actually accessed

The `mmap` feature is enabled by default. To disable it:

```toml
[dependencies]
onnx-ir = { version = "...", default-features = false }
```

## ONNX Compatibility

This library supports **all ONNX opset versions** (1 through 24) for every supported operator. Each
operator handles its full version history, including attribute-to-input migrations and
opset-dependent defaults. The opset compliance test suite verifies 461 operator-version combinations.

## Resources

- [Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md) -
  In-depth guide for adding new operators
- [Supported ONNX Operators](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md) -
  Full list of supported operators
- [Documentation](https://docs.rs/onnx-ir) - API documentation
