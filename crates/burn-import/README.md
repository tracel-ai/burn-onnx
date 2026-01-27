# burn-import (Legacy)

> **Deprecated:** This crate exists only for backward compatibility. Please use
> [`burn-onnx`](https://crates.io/crates/burn-onnx) directly.

## Migration Guide

ONNX import functionality has been moved from `burn-import` to `burn-onnx`.

### Update Your Dependencies

```toml
# Before
[build-dependencies]
burn-import = { version = "0.21", features = ["onnx"] }

# After
[build-dependencies]
burn-onnx = "0.21"
```

### Update Your Imports

```rust
// Before
use burn_import::onnx::ModelGen;

// After
use burn_onnx::ModelGen;
```

### Build Script Changes

```rust
// Before
use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("model.onnx")
        .out_dir("src/model/")
        .run();
}

// After
use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("model.onnx")
        .out_dir("src/model/")
        .run();
}
```

## Resources

- [burn-onnx Documentation](https://docs.rs/burn-onnx)
- [burn-onnx Repository](https://github.com/tracel-ai/burn-onnx)
- [Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md)
- [Supported ONNX Operators](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md)
