# ONNX to Burn: Development Guide

This guide offers in-depth design insights and step-by-step procedures for developers working on the
ONNX to Burn conversion tool. This tool allows the importation of ONNX models into the Burn deep
learning framework written in Rust. It converts ONNX models to Rust source code and model weights to
`.bpk` files.

For an introduction to ONNX import in Burn, see
[this section of the Burn book](https://burn.dev/books/burn/import/onnx-model.html).

## Design Overview

### Design Goals

- Perform best-effort conversion of ONNX models to Rust source code via Burn APIs.
- Convert ONNX model weights to Burn state files.
- Support ONNX models from all opset versions (1 through 24) for every supported operator.
- Produce easy-to-understand and modifiable models.
- Ensure the generated models are trainable using Burn APIs.

### Design Decisions

**Core Principles:**

- **Op/Node-Centric Design**: Built around individual operations and nodes for better scalability as
  more operators are added
- **Opset-Aware Processing**: Processors accept opset parameters for flexible behavior across
  different ONNX versions
- **Constants-First Approach**: All ONNX initializers are treated as constant nodes initially,
  providing a uniform starting point
- **Native Type Integration**: Direct use of `burn_tensor::TensorData` and `Dtype` for efficiency,
  consistency, and future mmap support
- **Multi-Phase Pipeline**: Explicit transformation phases (initialization → conversion → type
  inference → post-processing → finalization) for better visibility and maintainability
- **Graph Input Name Preservation**: Sanitized ONNX names are preserved for easier development and
  troubleshooting

**Separation of Concerns:**

- Limit interaction with ONNX to the Intermediate Representation (IR) stage to simplify the process
- Ensure operator behavior consistency across different OpSet versions
- Exclude any ONNX/Protobuf-specific logic from the Burn graph
- **Feature Support Validation**: The [`onnx-ir`](crates/onnx-ir/) crate should extract and preserve
  all ONNX attributes faithfully, even if Burn does not yet support them. Rejection of unsupported
  features should happen in [`burn-onnx`](crates/burn-onnx/) during code generation, not in
  `onnx-ir` during configuration extraction. This allows `onnx-ir` to be reused by other projects
  that may have different feature support
- **No `panic!` in codegen**: Structural validation (e.g., "only 1D/2D supported, got 3D") should
  use `ProcessError` in onnx-ir's `infer_types` or `extract_config`, not `panic!` in burn-onnx
  codegen. Panics in codegen produce poor error messages and crash the build process

The conversion process involves three main stages:

1. Convert ONNX model to Intermediate Representation (IR) via 5-phase pipeline.
2. Translate IR to a Burn graph.
3. Generate Rust source code from the Burn graph.

## Adding New Operators

To extend `burn-onnx` with support for new ONNX operators, follow these steps:

1. **Create PyTorch Script**: Place a PyTorch script using the new operator under
   `crates/onnx-tests/tests/<op>/<op>.py`. Make sure to print both input and output tensors for
   end-to-end testing.

2. **Generate ONNX Model**: Run the PyTorch script to produce an ONNX model.

3. **Visualize ONNX Model**: Use [Netron](https://github.com/lutzroeder/netron) to verify the ONNX
   model contains the expected operators.

4. **Generate IR and Burn Graph**: Run from the repository root:

   ```
   cargo run -p burn-onnx --bin onnx2burn -- crates/onnx-tests/tests/<op>/<op>.onnx ./out
   ```

5. **Implement Missing Operators**: If you encounter an error stating that an operator is
   unsupported, [implement it](#implementing-a-new-operator). The `./out/my-model.graph.txt` should
   provide relevant information.

6. **Inspect Generated Files**: The `my-model.graph.txt` contains IR details, `my-model.rs` holds
   the Burn model in Rust code, and `my-model.bpk` contains the model weights.

7. **Integration Test**: Include the test in the `tests/<op_name>/mod.rs` file in the
   [crates/onnx-tests/tests/](crates/onnx-tests/tests/) directory. Further details can be found in
   the [onnx-tests README](crates/onnx-tests/README.md).

## Implementing a New Operator

To extend the capabilities of the Burn library by supporting new operations imported from ONNX
graphs, developers must go through a few systematic steps. Here, we detail the process, using the
implementation of the `Squeeze` operation to illustrate points as needed. All file/directory paths
are relative to the root of the burn-onnx repository.

### Step 1: Node Processor Implementation in onnx-ir

The [`onnx-ir`](crates/onnx-ir/) crate handles the Intermediate Representation (IR) of ONNX models
using a processor-based architecture. For each operation:

1. **Create a node module** in `crates/onnx-ir/src/node/<operation_name>.rs`. This file should
   contain:
   - **Configuration struct**: Define operation-specific parameters (e.g., `SqueezeConfig`).
     **Important**: Include ALL ONNX operator attributes, even if burn-onnx doesn't use them yet.
     Use `Option<T>` for optional attributes.
   - **Processor struct**: Implement `NodeProcessor` trait (marked as `pub(crate)`)
   - The processor handles:
     - **Input/output specification**: Define expected inputs and outputs via `NodeSpec`
     - **Type inference**: Infer output types from inputs and configuration
     - **Configuration extraction**: Extract ALL operation parameters from ONNX attributes
     - **Node construction**: Build the final `Node` enum variant with config

2. **Make the module visible** in `crates/onnx-ir/src/node/mod.rs`:

   ```rust
   pub mod squeeze;
   ```

3. **Create a node struct** in your module file (e.g., `squeeze.rs`) with the standard fields:

   ```rust
   use onnx_ir_derive::NodeBuilder;

   #[derive(Debug, Clone, NodeBuilder)]
   pub struct SqueezeNode {
       pub name: String,
       pub inputs: Vec<Argument>,
       pub outputs: Vec<Argument>,
       pub config: SqueezeConfig,
   }
   ```

   The `NodeBuilder` derive macro generates a test builder (e.g., `SqueezeNodeBuilder`) with methods
   for constructing nodes in tests.

4. **Add to the macro invocation** in `crates/onnx-ir/src/ir/node.rs` by adding a mapping to the
   `define_node_enum!` macro:

   ```rust
   define_node_enum! {
       // ... other variants
       Squeeze => squeeze::SqueezeNode,
       // ... more variants
   }
   ```

   This single macro invocation generates both the `NodeType` enum (for parsing) and the `Node` enum
   (with tuple variants wrapping node structs) from a single source of truth.

5. **Register your processor** in `crates/onnx-ir/src/registry.rs` by adding it to the
   `with_standard_processors()` function:
   ```rust
   registry.register("Squeeze", Box::new(squeeze::SqueezeProcessor));
   ```

For example, the squeeze operation in `crates/onnx-ir/src/node/squeeze.rs` contains:

- A `SqueezeConfig` struct with operation parameters (axes)
- A `SqueezeProcessor` struct (marked `pub(crate)`) that implements `NodeProcessor`
- The `spec()` method defines input/output requirements
- The `build_node()` method extracts config and constructs the `Node::Squeeze` variant

### Step 2: Code Generation in burn-onnx

1. Create a new file named `<operation_name>.rs` in the `crates/burn-onnx/src/burn/node/` directory.
   This file implements code generation for your operation by implementing the `NodeCodegen` trait
   directly on the onnx-ir node type.

2. Implement the `NodeCodegen<PS>` trait for the onnx-ir node type. This trait defines how the node
   generates Rust code during the graph compilation process:

   ```rust
   use super::prelude::*;

   impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::squeeze::SqueezeNode {
       fn inputs(&self) -> &[Argument] {
           &self.inputs
       }

       fn outputs(&self) -> &[Argument] {
           &self.outputs
       }

       fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
           let input_arg = self.inputs.first().unwrap();
           let output_arg = self.outputs.first().unwrap();

           // Use scope.arg() to handle Tensor/Scalar/Shape arguments automatically
           let input = scope.arg(input_arg);
           let output = arg_to_ident(output_arg);

           // Access node configuration
           match &self.config.axes {
               Some(axes) => {
                   let axes_values: Vec<_> = axes.iter().map(|&i| {
                       proc_macro2::Literal::i64_suffixed(i)
                   }).collect();
                   quote! {
                       let #output = #input.squeeze_dims(&[#(#axes_values),*]);
                   }
               }
               None => {
                   // Get output rank from type inference
                   let output_rank = match &output_arg.ty {
                       ArgType::Tensor(t) => t.rank,
                       _ => panic!("Expected tensor output"),
                   };
                   quote! {
                       let #output = #input.squeeze::<#output_rank>();
                   }
               }
           }
       }
   }
   ```

   Key methods to implement:
   - `inputs(&self)` - Returns references to input arguments (usually just `&self.inputs`)
   - `outputs(&self)` - Returns references to output arguments (usually just `&self.outputs`)
   - `forward(&self, scope)` - Generates Rust code for the operation using the `quote!` macro
   - `field(&self)` - (Optional) Declares module fields for parameters like weights
   - `collect_snapshots(&self, field_name)` - (Optional) Collects tensor snapshots for burnpack
     serialization

3. Use helper utilities from `argument_helpers.rs`:
   - `scope.arg(argument)` - Automatically handles Tensor/Scalar/Shape with proper cloning
   - `arg_to_ident(argument)` - Converts argument to identifier for code generation

4. **Prefer existing Burn tensor APIs over manual loops**: Before implementing an operator with
   manual loops or per-element tensor operations in generated code, check the Burn tensor API for
   existing operations that do the same thing (e.g., `scatter` with `IndexingUpdateOp::Add`,
   `select_assign`, `unfold4d`). Native tensor operations are orders of magnitude faster than
   generated element-wise loops. When no exact Burn API exists, prefer compositions of tensor
   operations that stay on-device over approaches that move data between CPU and GPU (e.g.,
   `.into_data()` / `.from_data()`), as each transfer is a synchronization point that kills
   performance.

5. Add unit tests using snapshot testing to verify the generated code. These tests typically use the
   `insta` crate and test helper functions to validate the generated code:

   ```rust
   #[cfg(test)]
   mod tests {
       use super::super::test_helpers::*;
       use insta::assert_snapshot;
       use onnx_ir::squeeze::SqueezeNodeBuilder;

       #[test]
       fn test_squeeze_forward() {
           let node = SqueezeNodeBuilder::new("squeeze1")
               .input_tensor("input", 3, DType::F32)
               .output_tensor("output", 2, DType::F32)
               .axes(vec![1])
               .build();
           let code = codegen_forward_default(&node);
           assert_snapshot!(code, @"let output = input.squeeze_dims(&[1i64]);");
       }
   }
   ```

### Step 3: Register in Module System

Add the module declaration to `crates/burn-onnx/src/burn/node/mod.rs`:

```rust
// ... other node modules
pub(crate) mod squeeze;
// ... more node modules
```

The modules are automatically made visible through re-exports in the same file.

### Step 4: Register in Code Generation Dispatch

Add your operation to the dispatch macro in `crates/burn-onnx/src/burn/node_codegen.rs`. The
`impl_node_codegen_dispatch!` macro generates the trait implementation that dispatches to your
node-specific code.

Add the node variant name (as defined in `onnx-ir`'s `Node` enum) to the macro invocation:

```rust
impl_node_codegen_dispatch! {
    // ... other operations
    Squeeze,  // Add your operation here (matches Node::Squeeze variant)
    // ... more operations
}
```

The macro automatically generates:

- Dispatch implementation for `NodeCodegen<PS>` on `onnx_ir::Node`
- All required trait methods (`inputs`, `outputs`, `forward`, `field`, etc.)
- Pattern matching to route to your node-specific implementation

### Step 5: Processor Implementation

The `NodeProcessor` trait defines how operations are processed in onnx-ir. Each processor must
implement:

1. **Associated type**: `type Config` - Define your configuration struct (use `()` if no config)
2. **`infer_types()`** - Infer output types from inputs and config (required)
3. **`build_node()`** - Construct the node struct and wrap it in the `Node` enum variant (required)
4. **`extract_config()`** - Extract config from attributes/inputs (override if Config != `()`)
5. **`spec()`** - Define opset and input/output requirements (optional)
6. **`lift_constants()`** - Request constant lifting for inputs (optional)
7. **`is_noop()`** - Return `true` if the node is a no-op (optional, default `false`)

**Important**: Processors should extract the attributes they need and ignore the rest. Do not iterate
over all attributes to reject unknown ones, as ONNX may add new attributes in future opsets.

Example `build_node()` implementation:

```rust
fn build_node(&self, builder: RawNode, opset: usize) -> Node {
    let config = self.extract_config(&builder, opset).expect("Config extraction failed");
    Node::Squeeze(SqueezeNode {
        name: builder.name,
        inputs: builder.inputs,
        outputs: builder.outputs,
        config,
    })
}
```

Note: `RawNode` is the intermediate node representation used during processing. The `build_node()`
method converts it into the final typed `Node` enum variant.

For complete examples, see existing processors:

- **Simple operation**: `crates/onnx-ir/src/node/softmax.rs`
- **With constant inputs**: `crates/onnx-ir/src/node/squeeze.rs`
- **Complex operation**: `crates/onnx-ir/src/node/conv2d.rs`

See [NodeProcessor Trait](#nodeprocessor-trait) for the complete trait definition.

### Step 6: Add Newly Supported Op!

As a reward, add an extra check to `SUPPORTED-ONNX-OPS.md`!

### Constant Lifting

The onnx-ir pipeline automatically handles constant lifting during the post-processing phase.
"Lifting" constants means making constant values directly accessible on node inputs via
`Argument::value()`, instead of requiring a separate graph traversal to find a Constant node.

**When to use**: If your operation takes constant inputs (e.g., weights in Conv1d, shape tensors in
Reshape, axes in Squeeze), access them via `node.inputs[N].value()` in your `extract_config()`
method. See the [Configuration Extraction example](#example-configuration-extraction) in Step 5.

**Optional optimization**: Implement `lift_constants()` to explicitly request constant lifting for
specific inputs before `extract_config()` is called. The pipeline handles this automatically during
post-processing.

### Handling Optional ONNX Inputs

ONNX uses an empty string `""` for "optional input not provided". The pipeline sets
`ValueSource::Optional` on these inputs during parsing. Use `RawNode::get_input(index)` to access
inputs safely: it returns `None` for both out-of-bounds and optional inputs.

```rust
// Good: returns None for absent or optional inputs
if let Some(input) = node.get_input(2) {
    // input is guaranteed to be a real, non-optional input
    let value = input.value();
}

// Good: explicit is_optional() check when you need the index for mutation
if node.inputs.len() > 1 && !node.inputs[1].is_optional() && node.inputs[1].is_constant() {
    node.inputs[1].to_static()?;
}

// Bad: creates RuntimeInputRef with empty name, panics during codegen
if let Some(input) = node.inputs.get(2) {
    return Ok(SomeConfig::Runtime(RuntimeInputRef::new(input.name.clone(), 2)));
}

// Bad: unreliable after to_static() clears names
if !node.inputs[0].name.is_empty() { ... }
```

**Key rules:**

- Use `node.get_input(index)` in `extract_config()`, `is_noop()`, and read-only access
- Use `!node.inputs[N].is_optional()` guards in `lift_constants()` (which needs `&mut` access)
- Never check `name.is_empty()` to detect optional inputs; use `is_optional()` instead

## Architecture Overview

### ONNX-IR Pipeline

The [`onnx-ir`](crates/onnx-ir/) crate converts ONNX models to an Intermediate Representation
through a 5-phase pipeline:

#### Phase 1: Initialization

- Creates `GraphState` from ONNX proto structures
- **Constants-first approach**: Converts all ONNX initializers into Constant nodes, providing a
  uniform starting point for processing
- Sets up the value store for tensor data using `burn_tensor::TensorData`
- Preserves sanitized graph input names for debugging

#### Phase 2: Node Conversion

- Converts ONNX nodes to IR nodes using registered processors
- Creates `RawNode` instances from ONNX proto nodes (intermediate representation)
- Processors extract configuration and construct typed `Node` enum variants
- Handles constant nodes specially (extracting values from attributes into tensor store)
- Each processor is responsible for its own type inference and node construction

#### Phase 3: Type Inference

- Type inference happens within each processor's `process()` method during Phase 2
- Processors infer output types based on input types and configuration
- Multi-pass processing handles dependencies between nodes
- The pipeline may need multiple iterations for complex type dependencies (e.g., control flow)

#### Phase 4: Post-processing

- Lifts constants: Makes constant values accessible on downstream node inputs
- Eliminates no-op nodes: Removes nodes whose processor's `is_noop()` returns `true` (Identity,
  same-type Cast, scalar Reshape, scalar Gather, etc.) and rewires the graph
- Re-runs constant lifting after no-op elimination

#### Phase 4b: Simplification (Optional)

When `ModelGen::simplify(true)` is enabled, an additional simplification pass runs after
post-processing. This pass folds shape-related computations into constants at codegen time:

- **Shape folding**: `Shape(x)` with static dims becomes a constant array
- **Gather on shape**: `Gather(Shape(x), const_idx)` becomes a scalar constant
- **Slice on shape**: `Slice(Shape(x), start, end)` becomes a sub-array constant
- **Concat of shapes**: `Concat(Shape(x), Shape(y))` becomes a concatenated constant
- **Reshape from shape**: `Reshape(x, Shape(y))` uses a folded constant shape
- **Binary ops on shapes**: `Add/Mul/Sub/Div(Shape(x), const)` becomes a constant
- **Cast on shape**: `Cast(Shape(x))` becomes a constant
- **Where on shapes**: `Where(cond, Shape(x), Shape(y))` becomes a conditional constant
- **Expand from shape**: `Expand(x, Shape(y))` uses a folded constant shape
- **ConstantOfShape optimization**: `ConstantOfShape(Shape(x))` uses a known shape

Simplification is enabled by default. Existing operator tests explicitly use `.simplify(false)` when they need to test unsimplified codegen paths.
Use `--no-simplify` to disable it:

```sh
cargo run -p burn-onnx --bin onnx2burn -- model.onnx ./out             # simplification enabled (default)
cargo run -p burn-onnx --bin onnx2burn -- model.onnx ./out --no-simplify  # simplification disabled
```

Existing operator tests use `.simplify(false)` to test unsimplified codegen. Dedicated comparison
tests in `crates/onnx-tests/tests/simplify/` verify that simplified and unsimplified codegen produce
identical outputs.

#### Phase 5: Finalization

- Removes unreferenced constant nodes
- Constructs the final `OnnxGraph` with inputs, outputs, and nodes

### NodeProcessor Trait

The `NodeProcessor` trait (defined in `crates/onnx-ir/src/processor.rs`) is the core abstraction for
handling ONNX operations. Each processor implements:

**Required:**

- `type Config` - Associated type for configuration (use `()` if no config needed)
- `infer_types()` - Infer output types from inputs and configuration
- `build_node()` - Construct the final `Node` enum variant

**Static Shape Inference:**

The `infer_types()` method should always produce a `static_shape` (`Some(vec![...])`) on tensor
outputs, even when the input has no static shape info. Use `None` for unknown individual dimensions
and `Some(value)` for known ones. This enables downstream merging via `merge_static_shape()`.

- **Always produce `Some(vec![...])`**, never leave `static_shape` as `None` when you know the
  output rank. Start with `vec![None; rank]` and fill in whatever dimensions you can determine.
- **Extract dimension info from all available sources**: input `static_shape`, weight tensor data
  (via `node.inputs[N].value().map(|d| d.shape)`), weight `static_shape`, and operator config.
- **Use `unwrap_or_else`** instead of `.map()` on input static shape to avoid short-circuiting:
  ```rust
  // Good: always produces partial shape
  let mut shape = tensor.static_shape.clone()
      .unwrap_or_else(|| vec![None; tensor.rank]);
  shape[1] = Some(out_channels); // fill in what we know
  Some(shape)

  // Bad: returns None entirely when input has no static_shape
  tensor.static_shape.as_ref().map(|s| { ... })
  ```

**Optional (have defaults):**

- `spec()` - Define opset requirements and input/output count validation (`NodeSpec`, `InputSpec`,
  `OutputSpec`)
- `extract_config()` - Extract configuration from attributes/inputs (default returns
  `Default::default()`)
- `lift_constants()` - Request constant lifting for specific inputs (default does nothing)
- `input_preferences()` - Declare preferred input types from producers (default returns `None`)
- `is_noop()` - Return `true` if this node is a no-op after type inference, causing it to be
  eliminated during post-processing (default returns `false`)

Design principles: Each processor is self-contained, handling type inference, config extraction, and
node construction. Processors return strongly-typed `Node` enum variants, ensuring type safety
throughout the pipeline.

**Error formatting**: `ProcessError` has a `Display` impl for user-facing messages. Use `{}` (not
`{:?}`) when formatting errors to avoid exposing Rust variant names like `Custom("...")` to users.

## Testing

When implementing a new operator, there are several levels of testing to consider:

### Unit Testing

- **Processor Methods**: Write unit tests in `crates/onnx-ir/src/node/<operation_name>.rs` to
  verify:
  - `extract_config()` - Correctly extracts configuration from attributes and inputs
  - `infer_types()` - Correctly infers output types (element type, rank, static shapes)
  - `build_node()` - Constructs correct `Node` enum variant
  - `spec()` - Defines correct opset and input/output requirements
  - Error handling for invalid inputs or configurations

  See existing tests in `crates/onnx-ir/src/node/squeeze.rs` for examples.

- **Code Generation**: Test the burn-onnx Node implementation to verify correct Rust code
  generation. Use `insta` snapshot tests to cover as many code generation branches as possible:
  - Each configuration option (e.g., different axis values, padding modes)
  - Each input type variant (tensor, scalar, shape)
  - Optional vs required inputs
  - Different tensor ranks and data types
  - Edge cases that trigger different code paths
  - **Use inline snapshots only**: Use `assert_snapshot!(code, @r"...")` with embedded expected
    output, not external `.snap` files

### Integration Testing

Integration tests are located in [`crates/onnx-tests/`](crates/onnx-tests/) and verify end-to-end
conversion from ONNX models to Burn source code.

#### Directory Structure

- `tests/<op_name>/`: Each operator has its own directory
- `tests/<op_name>/<op_name>.py`: Python script that generates the ONNX model
- `tests/<op_name>/<op_name>.onnx`: Generated ONNX model
- `tests/<op_name>/mod.rs`: Test implementation for the operator

#### Python Script Format

Use [`uv`](https://docs.astral.sh/uv/) inline script format for self-contained test scripts:

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
# ... rest of script
```

This makes scripts executable without manual environment setup.

#### Creating a Test for a New Operator

There are two approaches to generating ONNX files:

**Approach 1: Exporting a PyTorch Model** (most common)

```python
#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "torch==2.1.1",
# ]
# ///

import torch
import torch.nn as nn
from onnx.reference import ReferenceEvaluator
import onnx

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return my_operation(x)

model = MyModel()
torch.manual_seed(42)
input_tensor = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    input_tensor,
    "tests/my_new_op/my_new_op.onnx",
    opset_version=16,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=False  # Preserve operators
)

# Verify with ONNX ReferenceEvaluator (ground truth)
onnx_model = onnx.load("tests/my_new_op/my_new_op.onnx")
ref = ReferenceEvaluator(onnx_model)
outputs = ref.run(None, {"input": input_tensor.numpy()})

print("Input:", input_tensor)
print("Expected output:", outputs[0])
```

**Approach 2: Constructing ONNX Graph Directly**

Useful when you need precise control over operator attributes:

```python
#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
data = np.random.randn(5, 5, 5).astype(np.float32)
indices = np.array([0, 2, 4], dtype=np.int64)

node = helper.make_node("Gather", inputs=["data", "indices"], outputs=["output"], axis=1)

data_tensor = helper.make_tensor_value_info("data", TensorProto.FLOAT, data.shape)
indices_tensor = helper.make_tensor_value_info("indices", TensorProto.INT64, indices.shape)
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [5, 3, 5])

graph = helper.make_graph([node], "gather-model", [data_tensor, indices_tensor], [output_tensor])
model = helper.make_model(graph)
onnx.save(model, "tests/my_new_op/my_new_op.onnx")

# Verify with ONNX ReferenceEvaluator (ground truth)
ref = ReferenceEvaluator(model)
outputs = ref.run(None, {"data": data, "indices": indices})

print("Data:", data)
print("Indices:", indices)
print("Expected output:", outputs[0])
```

#### Using ReferenceEvaluator

Always use `onnx.reference.ReferenceEvaluator` to compute expected outputs. This is the official
ONNX reference implementation and serves as ground truth for verifying Burn's output matches.

#### Creating the Rust Test

Create `tests/my_new_op/mod.rs`:

```rust
use crate::include_models;
include_models!(my_new_op);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use crate::backend::TestBackend;

    #[test]
    fn my_new_op() {
        let device = Default::default();
        let model = my_new_op::Model::<TestBackend>::new(&device);

        let input = Tensor::ones([1, 3, 224, 224], &device);
        let output = model.forward(input);

        // Compare against expected values from ReferenceEvaluator
        let expected_shape = Shape::from([1, 3, 224, 224]);
        assert_eq!(expected_shape, output.shape());
    }
}
```

Register the test module in `tests/test_mod.rs`:

```rust
pub mod my_new_op;
```

#### Running Tests

```sh
# Default backend (NdArray)
cargo test

# WGPU backend
cargo test --features test-wgpu

# LibTorch backend
cargo test --features test-tch

# Specific test
cargo test --test test_mod my_new_op::test_my_new_op
```

#### Best Practices

**Model Generation:**

- Keep models simple, focusing on single operators
- Use fixed seeds for reproducibility: `torch.manual_seed(42)`
- Print input/output tensors for reference
- Verify with [Netron](https://github.com/lutzroeder/netron)
- Use `do_constant_folding=False` if PyTorch optimizes away operators

**Test Implementation:**

- Test multiple input shapes, data types, and parameters
- Include edge cases (empty tensors, single elements, large tensors)
- Use appropriate numerical tolerance levels
- Test error cases for invalid inputs
- Cover at least one non-default configuration (e.g., non-unit strides, padding, dilation) in
  addition to the basic case, to exercise the major codegen branches

#### Debugging Failed Tests

1. **Inspect ONNX Model**: Use Netron to visualize structure
2. **Check Values**: Add print statements in Python scripts
3. **Generate Rust Code**: `cargo run -p burn-onnx --bin onnx2burn -- tests/my_op/my_op.onnx ./out`
4. **Numerical Issues**: Adjust tolerance for precision problems

Testing the processor implementation is particularly important as it directly affects the
correctness of the conversion process. Incorrect type inference can lead to mismatched tensor shapes
or wrong element types, while incorrect configuration extraction can cause runtime errors or produce
incorrect results.

## Node Enum Architecture

The ONNX-IR uses an enum-based node representation where each ONNX operation is a variant of the
`Node` enum (defined in `crates/onnx-ir/src/ir/node.rs`). Each variant wraps an operation-specific
node struct (e.g., `SoftmaxNode`, `Conv2dNode`) that contains `name`, `inputs`, `outputs`, and
optionally a `config` field.

The `define_node_enum!` macro generates both enums from a single source using the syntax
`VariantName => module::NodeStructType`:

```rust
define_node_enum! {
    Softmax => softmax::SoftmaxNode,
    Conv2d => conv2d::Conv2dNode,
    Squeeze => squeeze::SqueezeNode,
    // ... 200+ more variants
}
```

This macro generates:

1. **`NodeType` enum**: Simple unit variants for ONNX parsing (`Softmax`, `Conv2d`, etc.)
2. **`Node` enum**: Tuple variants wrapping node structs (`Softmax(SoftmaxNode)`,
   `Conv2d(Conv2dNode)`, etc.)
3. **Accessor methods**: `name()`, `inputs()`, `outputs()` automatically generated for the `Node`
   enum

This design provides:

- **Type safety**: Each operation has its own struct type
- **Trait implementations**: Operations can implement specific traits on their node structs
- **Single source of truth**: Both enums are guaranteed to stay in sync
- **Pattern matching**: Easy to match on specific operations and access their configuration

## Model Checks

The `crates/model-checks/` directory contains real-world model validation tests. Each subdirectory is
a standalone crate that downloads a model, generates Burn code from it, and runs inference to verify
correctness.

### Running Model Checks

Use the `model-check` xtask command:

```sh
# Run all models (download + build + run, ndarray backend, release mode)
cargo xtask model-check

# Single model
cargo xtask model-check --model silero-vad

# Subcommands: download, build, run, all (default)
cargo xtask model-check --model silero-vad build

# Select backend
cargo xtask model-check --features tch

# Debug build
cargo xtask model-check --model silero-vad --debug

# Stop on first failure (default: continue and report summary)
cargo xtask model-check --fail-fast
```

### Available Models

| Model | Directory | Notes |
|---|---|---|
| Silero VAD | `silero-vad` | Voice activity detection |
| all-MiniLM-L6-v2 | `all-minilm-l6-v2` | Sentence embeddings |
| CLIP ViT-B-32 text | `clip-vit-b-32-text` | Text encoder |
| CLIP ViT-B-32 vision | `clip-vit-b-32-vision` | Vision encoder |
| ModernBERT-base | `modernbert-base` | Language model |
| RF-DETR Small | `rf-detr` | Object detection |
| ALBERT | `albert` | Language model (requires Python 3.11) |
| YOLO v8n | `yolo` | Object detection |
| MediaPipe Face Detector | `mediapipe-face-detector` | Face detection |

### Model Artifacts

Model artifacts (ONNX files, test data) are stored in the platform cache directory:

- macOS: `~/Library/Caches/burn-onnx/model-checks/<model-name>/`
- Linux: `~/.cache/burn-onnx/model-checks/<model-name>/`

Set `BURN_CACHE_DIR` to override the base cache path (useful for CI).

### Adding a New Model Check

1. Create a new directory under `crates/model-checks/<model-name>/`
2. Add `Cargo.toml` with `[workspace]` (standalone), `burn` and `burn-store` dependencies, and
   backend feature flags forwarding to `burn/<backend>`
3. Add `get_model.py` (uv script format) to download and prepare the ONNX model
4. Add `build.rs` to generate Burn code from the ONNX model via `ModelGen`
5. Add `src/main.rs` to load the model, run inference, and compare against reference outputs
6. Register the model in `xtask/src/model_check.rs` in the `MODELS` array

## Resources

1. [PyTorch to ONNX](https://pytorch.org/docs/stable/onnx.html)
2. [ONNX to PyTorch](https://github.com/ENOT-AutoDL/onnx2torch)
3. [ONNX Introduction](https://onnx.ai/onnx/intro/)
4. [ONNX Operators](https://onnx.ai/onnx/operators/index.html)
5. [ONNX Protos](https://onnx.ai/onnx/api/classes.html)
6. [ONNX Optimizer](https://github.com/onnx/optimizer)
7. [Netron](https://github.com/lutzroeder/netron)
