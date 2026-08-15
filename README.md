<div align="center">

# Burn ONNX

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-onnx.svg)](https://crates.io/crates/burn-onnx)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/burn-onnx)
[![Test Status](https://github.com/tracel-ai/burn-onnx/actions/workflows/test.yml/badge.svg)](https://github.com/tracel-ai/burn-onnx/actions/workflows/test.yml)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tracel-ai/burn-onnx)

**Import ONNX models into the [Burn](https://burn.dev) deep learning framework.**

[Repository](https://github.com/tracel-ai/burn-onnx) |
[Burn Repository](https://github.com/tracel-ai/burn)

</div>

## Overview

`burn-onnx` converts ONNX models to native Burn Rust code, allowing you to run models from PyTorch,
TensorFlow, and other frameworks on any Burn backend - from WebAssembly to CUDA.

**Key features:**

- Generates readable, modifiable Rust source code from ONNX models
- Produces `burnpack` weight files for efficient loading
- Works with any Burn backend (CPU, GPU, WebGPU, embedded)
- Supports both `std` and `no_std` environments
- Full opset compliance: all supported operators work across ONNX opset versions 1 through 24
- Extensible: hook operators outside the supported set (custom/vendor domains) to your own Rust
  kernels, or override the generated code for a built-in operator
- Graph simplification (enabled by default): attention coalescing, constant folding, constant shape
  propagation, idempotent-op elimination, identity-element elimination, CSE, dead code elimination,
  and permute-reshape detection

## Quick Start

Add to your `Cargo.toml`:

```toml
[build-dependencies]
burn-onnx = "0.21"
```

In your `build.rs`:

```rust
use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/my_model.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

Include the generated code in `src/model/mod.rs`:

```rust
pub mod my_model {
    include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
}
```

Then use the model:

```rust
use crate::model::my_model::Model;

let model: Model = Model::default();
let output = model.forward(input_tensor);
```

For detailed usage instructions, see the
[ONNX Import Guide](https://burn.dev/books/burn/onnx-import.html) in the Burn Book.

## Custom Operators

Operators outside the [supported set](SUPPORTED-ONNX-OPS.md) do not have to block an import. That
covers vendor domains such as `com.microsoft`, ops from a framework's custom export, and op types
not implemented yet. Register a hook and supply the Rust yourself:

```rust
// build.rs
ModelGen::new()
    .input("src/model/my_model.onnx")
    .out_dir("model/")
    .register_custom_op(FftReal)      // handles my_domain::FftReal
    .register_op_override(MyMatMul)   // replaces the generated code for every MatMul
    .run_from_script();
```

- **`CustomOp`** supplies type inference and code generation for one ONNX `(op_type, domain)`. It
  can read the node's attributes and its constant inputs.
- **`OpOverride`** replaces the generated code for a built-in operator, to route it to a fused,
  quantized, or hardware-specific kernel of your own. Type inference still comes from the built-in.

Everything a hook needs is re-exported from `burn_onnx::ext`, so implementing one does not mean
depending on `onnx-ir` or matching `proc-macro2`/`quote` versions by hand.

Not sure which operators a given model needs? Run the import with no hooks registered: it fails with
a list of every unsupported operator, its domain, and how many nodes use it.

Runnable example: [custom-op-hooks](examples/custom-op-hooks). Full reference: "Custom Operators and
Overrides" in the [Development Guide](DEVELOPMENT-GUIDE.md).

## Examples

| Example                                                       | Description                                    |
| ------------------------------------------------------------- | ---------------------------------------------- |
| [onnx-inference](examples/onnx-inference)                     | Basic ONNX model inference                     |
| [image-classification-web](examples/image-classification-web) | WebAssembly/WebGPU image classifier            |
| [custom-op-hooks](examples/custom-op-hooks)                   | Hooks for custom ops and built-in op overrides |

## Model Validation

We validate burn-onnx against
[26 real-world models](https://github.com/tracel-ai/burn-onnx/tree/main/crates/model-checks)
spanning image classification, object detection, depth estimation, NLP, speech, and generative AI.
Each model check verifies the full pipeline: ONNX import, Rust codegen, weight loading, and
numerical accuracy against ONNX Runtime reference outputs.

## Supported Operators

See the [Supported ONNX Operators](SUPPORTED-ONNX-OPS.md) table for the complete list of supported
operators. Anything outside that table can still be imported by registering a hook — see
[Custom Operators](#custom-operators) above.

## Contributing

We welcome contributions! Please read the [Contributing Guidelines](CONTRIBUTING.md) before opening
a PR, and the [Development Guide](DEVELOPMENT-GUIDE.md) for architecture and implementation details.

For questions and discussions, join us on [Discord](https://discord.gg/uPEBbYYDB6).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
