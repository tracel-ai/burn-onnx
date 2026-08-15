# Custom Op Hooks

Importing an ONNX model that contains operators `burn-onnx` does not know, plus replacing the
generated code for a built-in operator with your own kernel.

The model in `src/model/custom_model.onnx` stands in for the common real-world case: an export where
some layers came out as ops in a vendor domain.

```
x [2,4] ──MatMul(w)──▶ example.custom::ScaleBias ──Sigmoid──▶ example.custom::ChannelScale ──▶ y [2,3]
             built-in         custom op (attrs)      built-in        custom op (constant input)
                                                     overridden
```

Three hooks in [`build.rs`](build.rs) cover it:

| Hook              | Kind         | Demonstrates                                                           |
| ----------------- | ------------ | ---------------------------------------------------------------------- |
| `ScaleBias`       | `CustomOp`   | reading ONNX attributes; parsing them once for both hook methods       |
| `ChannelScale`    | `CustomOp`   | reading a **constant initializer input** at build time and inlining it |
| `SigmoidOverride` | `OpOverride` | replacing a built-in operator's generated code                         |

The kernels the hooks call live in [`src/ops.rs`](src/ops.rs) — ordinary Rust functions you can
test, benchmark, or swap out without touching the import.

## Usage

```bash
cargo run -p custom-op-hooks --bin custom_op_demo
```

Output:

```text
output: Tensor {
  data:
[[1.0, 0.27318323, 0.0067903423],
 [1.376118e-9, 0.3464806, 1.9974966]],
  shape:  [2, 3],
  ...
}
matches the ONNX reference
```

The binary asserts against the values in `src/model/generate_model.py`, so a passing run means the
whole path — parse, hook type inference, codegen, weight loading, execution — is correct.

To regenerate the ONNX file (and print the reference values):

```bash
cd src/model && uv run generate_model.py
```

## What gets generated

`build.rs` writes `$OUT_DIR/model/custom_model.rs`. Its `forward` is where the hooks landed:

```rust
pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
    let constant2_out1 = self.constant2.val();          // see "inlined constants" below
    let linear1_out1 = self.linear1.forward(x);         // built-in
    let custom1_out1 = ops::scale_bias(linear1_out1, 2f32, 0.5f32);        // ScaleBias hook
    let sigmoid1_out1 = ops::fast_sigmoid(custom1_out1);                   // Sigmoid override
    let custom2_out1 = ops::channel_scale(
        sigmoid1_out1,
        &[1f32, 0.5f32, 2f32],                          // inlined at build time
        &self.device,
    );
    custom2_out1
}
```

Reading this file is the fastest way to debug a hook: compile errors from your emitted tokens point
straight into it.

## Things worth knowing

**Discovering which hooks a model needs.** Register none and build. The failure lists every custom
`(domain, op_type)` pair with how many nodes use it — that list is your TODO:

```text
Failed to parse ONNX file 'src/model/custom_model.onnx': model contains 2 custom op(s) with no
covering inference hook:
  - example.custom::ChannelScale used by 1 node(s)
  - example.custom::ScaleBias used by 1 node(s)
Register hooks via ModelGen::register_custom_op.
```

**Override targets are IR node types, not ONNX op types.** This model's `MatMul` never reaches
codegen as a `MatMul`: because its weight is a constant 2D initializer, the parser rewrites it to
`Linear`. An `OpOverride` for `NodeType::MatMul` would silently never fire. Check the generated code
(or `ModelGen::development(true)`, which dumps the parsed graph) to see the node types your override
will actually see.

**Emitted paths resolve in _your_ crate.** The generated file is `include!`-ed into this crate, so
the hooks emit `crate::ops::...` (registered as `use crate::ops;`). A path like `my_crate::ops::...`
only works if `my_crate` is a dependency.

**Inlined constants leave an unused binding.** `ChannelScale` reads its constant input at build time
and inlines the values, so nothing consumes that input at runtime — but the initializer is still
lifted to a model parameter, producing the unused `constant2_out1` line above (and an
`unused_variables` warning). It is harmless. The alternative is to consume the input with
`ctx.arg(&node.inputs[1])`, which uses the runtime tensor from the weights file instead of inlining.

**Loading weights.** This model has real parameters (the `Linear` weights), so the demo uses
`Model::from_file(...)`. `Model::new(&device)` would leave them zeroed.

## Writing your own hooks

See the "Custom Operators and Overrides" section of
[DEVELOPMENT-GUIDE.md](../../DEVELOPMENT-GUIDE.md) for the full reference: the trait methods, the
`ArgType` → generated-Rust-type mapping, the output-binding convention, and the iteration workflow.
