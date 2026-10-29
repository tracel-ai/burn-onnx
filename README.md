<div align="center">

# Burn ONNX

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-onnx.svg)](https://crates.io/crates/burn-onnx)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/burn-onnx)
[![Test Status](https://github.com/tracel-ai/burn-onnx/actions/workflows/test.yml/badge.svg)](https://github.com/tracel-ai/burn-onnx/actions/workflows/test.yml)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-onnx/blob/main/LICENSE-MIT)
[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-blue)](https://deepwiki.com/tracel-ai/burn-onnx)

**Turn ONNX models into native Rust code for the [Burn](https://burn.dev) deep learning
framework.**

[Docs](https://docs.rs/burn-onnx) | [Burn Book](https://burn.dev/books/burn/onnx-import.html) |
[Supported Operators](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md) |
[Examples](https://github.com/tracel-ai/burn-onnx/tree/main/examples) |
[Discord](https://discord.gg/uPEBbYYDB6)

</div>

`burn-onnx` reads an ONNX model exported from PyTorch, TensorFlow, JAX, or anything else that speaks
ONNX, and writes it out as plain Burn code: a `Model` struct with a typed `forward` method, plus a
weights file. There is no ONNX runtime to ship. The result compiles with the rest of your crate and
runs on every Burn backend, from a browser tab or a microcontroller to a CUDA GPU.

- **Readable output.** The generated `.rs` file is ordinary Burn code you can read, step through in a
  debugger, and edit.
- **Any backend, any target.** CPU, CUDA, Metal, Vulkan, WebGPU, WebAssembly, and `no_std` embedded.
- **Every opset.** Each supported operator handles ONNX opsets 1 through 24, including the attribute
  to input migrations and changed defaults along the way.
- **Optimized at import.** Constant folding, shape propagation, common subexpression elimination,
  dead code removal, and attention fusion run before code generation.
- **Built for big models.** Large graphs are split into submodules so the generated code stays
  compilable, which takes graphs as large as the 28,000-node Stable Diffusion XL UNet.
- **Extensible.** Operators outside the supported set, including vendor domains like
  `com.microsoft`, can be implemented with your own Rust instead of blocking the import.
- **Trainable.** The imported model is a regular Burn `Module`, so you can fine-tune it.
- **Export too.** Experimental support for the reverse direction: save a Burn module as an ONNX
  file.

## Quick Start

Add the dependencies to `Cargo.toml`. The generated code loads its weights through `burn-store`, so
your crate needs it as well, and `burn` needs a backend feature (`flex` is the portable CPU
backend):

```toml
[dependencies]
burn = { version = "0.22", features = ["flex"] }
burn-store = "0.22"

[build-dependencies]
burn-onnx = "0.22"
```

Convert the model in `build.rs`:

```rust
use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/my_model.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

Include the generated code, for example from `src/model/mod.rs`:

```rust
pub mod my_model {
    include!(concat!(env!("OUT_DIR"), "/model/my_model.rs"));
}
```

Then run it:

```rust
use burn::tensor::{Device, Tensor};
use crate::model::my_model::Model;

fn main() {
    let device = Device::default();
    let model = Model::default(); // loads the weights written by build.rs
    let input = Tensor::<4>::zeros([1, 3, 224, 224], &device);
    let output = model.forward(input);
    println!("{output}");
}
```

The [Burn Book chapter](https://burn.dev/books/burn/onnx-import.html) walks through the same steps
in more detail, and [onnx-inference](https://github.com/tracel-ai/burn-onnx/tree/main/examples/onnx-inference)
is a complete project that classifies MNIST digits.

### What gets generated

For each `.onnx` input, `ModelGen` writes a `.rs` file and a `.bpk` (Burnpack) weights file. Here
is the `forward` method generated for the MNIST model in the example above:

```rust
pub fn forward(&self, input_1: Tensor<4>) -> Tensor<2> {
    let conv2d1_out1 = self.conv2d1.forward(input_1);
    let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
    let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
    let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
    // ...
    let linear2_out1 = self.linear2.forward(relu4_out1);
    let batchnormalization2_out1 = self.batchnormalization2.forward(linear2_out1);
    let logsoftmax1_out1 = log_softmax(batchnormalization2_out1, 1);
    logsoftmax1_out1
}
```

Layers with weights become Burn modules (`Conv2d`, `Linear`, `BatchNorm`, ...) and everything else
becomes direct tensor operations. Pass `.development(true)` to `ModelGen` to also dump the parsed
ONNX graph next to the code, which helps when debugging an import.

### Loading weights

`ModelGen::load_strategy` decides how the model finds its weights at runtime:

| `LoadStrategy`     | Generated constructors                          | Good for                                 |
| ------------------ | ----------------------------------------------- | ---------------------------------------- |
| `File` (default)   | `default()`, `from_file(path, &device)`, `from_bytes(..)` | desktop and server apps        |
| `Embedded`         | `default()`, `from_embedded(&device)`, `from_bytes(..)`   | single-binary deploys, embedded |
| `Bytes`            | `from_bytes(bytes, &device)`                    | WebAssembly, custom loaders              |
| `None`             | none                                            | managing weights yourself                |

With `File`, `Model::default()` reads the `.bpk` from the path it was written to at generation time:
an absolute path inside `OUT_DIR` from a build script, or the `out_dir` exactly as given to
`onnx2burn` or `run_from_cli`, which a relative path resolves against the working directory at
runtime. That is convenient during development; for a binary you distribute, ship the `.bpk` and
call `Model::from_file(path, &device)`, or switch to `Embedded`. Avoid `Model::new(&device)` on its
own: it builds the structure without loading any weights.

## Command Line

`onnx2burn` runs the conversion outside a build script. It is useful for reading the generated code,
or for checking in generated code that you intend to edit by hand:

```sh
cargo install burn-onnx
onnx2burn my_model.onnx ./generated
```

Flags: `--embed-states` (embed weights in the code), `--no-simplify`, `--no-partition`, and
`--no-development` (skip the debug dumps the CLI writes by default).

## Custom Operators

Operators outside the
[supported set](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md) do not have
to block an import: vendor domains such as `com.microsoft`, ops from a framework's custom export, and
op types not implemented yet. Register a hook and supply the Rust yourself:

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
  can read the node's attributes and constant inputs.
- **`OpOverride`** replaces the generated code for a built-in operator, to route it to a fused,
  quantized, or hardware-specific kernel of your own. Type inference still comes from the built-in.

Everything a hook needs is re-exported from `burn_onnx::ext`, so you never depend on `onnx-ir` or
match `proc-macro2`/`quote` versions by hand.

Not sure which operators a model needs? Build with no hooks registered. The import fails with a list
of every unsupported operator, its domain, and how many nodes use it:

```text
Failed to parse ONNX file 'src/model/custom_model.onnx': model contains 2 custom op(s) with no
covering inference hook:
  - example.custom::ChannelScale used by 1 node(s)
  - example.custom::ScaleBias used by 1 node(s)
Register hooks via ModelGen::register_custom_op.
```

The [custom-op-hooks](https://github.com/tracel-ai/burn-onnx/tree/main/examples/custom-op-hooks)
example is a working walkthrough, and the
[Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md) has the
full reference.

## Exporting to ONNX (experimental)

The `export` feature goes the other way. `OnnxExporter` runs a module's forward pass once on a
capture device, records the operations, and writes an ONNX model with the weights embedded:

```toml
[dependencies]
burn-onnx = { version = "0.22", features = ["export"] }
```

```rust
use burn_onnx::export::OnnxExporter;

let sample = Tensor::<4>::zeros([1, 3, 224, 224], &device);
OnnxExporter::new()
    .export(&model, sample, MyModel::forward)?
    .save("my_model.onnx")?;
```

`export` fixes every dimension to the sample's shape. `export_dynamic` takes a second sample and an
`InputSpec` per input to mark axes such as the batch size as symbolic. The exporter targets opset 18
and currently covers the operations typical of convolutional and fully connected networks
(ResNet-18 exports and passes the ONNX checker); anything it cannot lower yet is reported as
`ExportError::UnsupportedOperation`. See the
[`export` module docs](https://docs.rs/burn-onnx/latest/burn_onnx/export/index.html) for details.

## How It Works

```text
 model.onnx
     │
     ▼
 onnx-ir      parse protobuf ─▶ typed IR nodes ─▶ type & shape inference ─▶ simplification
     │
     ▼
 burn-onnx    Burn code generation ─▶ partition large graphs into submodules
     │
     ├─▶ model.rs    Model struct + forward(), formatted Rust
     └─▶ model.bpk   weights in Burnpack format
```

[`onnx-ir`](https://github.com/tracel-ai/burn-onnx/tree/main/crates/onnx-ir) is a standalone ONNX
parser that knows nothing about Burn: it turns the protobuf into a typed graph, parses each
operator's attributes into a typed config, infers types and static shapes, and simplifies the
graph. `burn-onnx` then maps each node
to Burn code. The [Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md)
covers each phase in detail.

## Crates

| Crate                                                     | Description                                                 |
| --------------------------------------------------------- | ----------------------------------------------------------- |
| [`burn-onnx`](https://crates.io/crates/burn-onnx)         | Code generator, `onnx2burn` CLI, and ONNX exporter          |
| [`onnx-ir`](https://crates.io/crates/onnx-ir)             | Framework-independent ONNX parser and intermediate representation |
| [`onnx-ir-derive`](https://crates.io/crates/onnx-ir-derive) | Derive macros used by `onnx-ir`                           |

## Examples

| Example                                                                                                      | Description                                             |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| [onnx-inference](https://github.com/tracel-ai/burn-onnx/tree/main/examples/onnx-inference)                   | MNIST classifier: the smallest complete import          |
| [image-classification-web](https://github.com/tracel-ai/burn-onnx/tree/main/examples/image-classification-web) | SqueezeNet in the browser with WebAssembly and WebGPU  |
| [raspberry-pi-pico](https://github.com/tracel-ai/burn-onnx/tree/main/examples/raspberry-pi-pico)             | `no_std` inference on a microcontroller, weights embedded |
| [custom-op-hooks](https://github.com/tracel-ai/burn-onnx/tree/main/examples/custom-op-hooks)                 | Custom operators and built-in operator overrides        |

## Testing and Compatibility

- **Real-world models.** [27 models](https://github.com/tracel-ai/burn-onnx/tree/main/crates/model-checks)
  are imported, compiled, and compared against ONNX Runtime outputs, covering image classification,
  detection, depth estimation, language models, speech, text-to-speech, and diffusion: ResNet,
  YOLO, CLIP, BERT variants, Qwen, SmolLM, Kokoro, Silero VAD, Depth Pro, Stable Diffusion XL, and
  more.
- **The official ONNX test suite.** 1,185 of the 1,765 upstream
  [ONNX backend node tests](https://github.com/tracel-ai/burn-onnx/tree/main/crates/onnx-official-tests)
  pass end to end on every CI run, with the status of the rest tracked in a checked-in expectations
  file.
- **Opset compliance.** Every supported operator is tested at every opset version it exists in, 461
  operator-version combinations in all.
- **Operator tests.** Hundreds of integration tests built from PyTorch- or NumPy-generated models,
  with expected outputs from the ONNX reference evaluator.

## Troubleshooting

- **Unsupported operator.** The error lists each missing operator. Check the
  [supported operators](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md),
  then either implement it as a [custom op](#custom-operators) or open an issue.
- **Generated code does not compile.** Make sure `burn`, `burn-store`, and `burn-onnx` are the same
  version. Find the generated file under `target/<profile>/build/<your-crate>-*/out/` and read it, or
  generate it with `onnx2burn` to inspect it directly. Please report it, with the model if you can.
- **Wrong outputs.** Confirm the model was loaded with `default()`, `from_file`, `from_embedded`, or
  `from_bytes`, not `new`. Then compare against ONNX Runtime with the same input.
- **Very old models.** Opsets 1 through 24 are supported, but upgrading an old model with
  [`onnx_opset_upgrade.py`](https://github.com/tracel-ai/burn-onnx/blob/main/onnx_opset_upgrade.py)
  (`uv run --script onnx_opset_upgrade.py`) to opset 16 also runs ONNX shape inference, which can
  help with models that carry little shape information.

## Contributing

Contributions are welcome. Please read the
[Contributing Guidelines](https://github.com/tracel-ai/burn-onnx/blob/main/CONTRIBUTING.md) before
opening a PR, and the [Development Guide](https://github.com/tracel-ai/burn-onnx/blob/main/DEVELOPMENT-GUIDE.md)
for the architecture and a step-by-step walkthrough of adding an operator. For questions and
discussion, join us on [Discord](https://discord.gg/uPEBbYYDB6).

## License

Licensed under either of
[Apache License, Version 2.0](https://github.com/tracel-ai/burn-onnx/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/tracel-ai/burn-onnx/blob/main/LICENSE-MIT) at your option.
