# ONNX Simplifier - Feature Tracking

Tracking issue: [#61](https://github.com/tracel-ai/burn-onnx/issues/61)

Reference: [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

## Design

- Implement as a `simplify` module within `onnx-ir` (not a separate crate)
- Add `simplify` flag to `OnnxGraphBuilder` (default: `true`)
- Expose `simplify` flag in `ModelGen` (default: `true`)
- Disable simplification in `onnx-tests` and `onnx-ir` integration tests

## Core Simplification Passes

- [ ] **Constant folding** - Evaluate operations with all-constant inputs at parse time, replace with constant tensors
- [ ] **Identity node elimination** - Remove identity/no-op nodes (e.g., `Identity`, reshape to same shape, transpose with identity perm)
- [ ] **Dead node elimination** - Remove nodes whose outputs are unused
- [ ] **Redundant node elimination** - Merge duplicate nodes that compute the same thing

## Fusion Passes

- [ ] **BatchNorm into Conv fusion** - Fold batch normalization parameters into convolution weights/bias
- [ ] **Pad into Conv/Pool fusion** - Merge explicit Pad nodes into the padding attribute of Conv/Pool ops

## Shape and Type Simplification

- [ ] **Shape inference propagation** - Iteratively infer and propagate static shapes through the graph
- [ ] **Dynamic-to-static shape resolution** - Replace dynamic dimensions with static values when input shapes are known

## Graph Cleanup

- [ ] **Unused output elimination** - Remove graph outputs that are not needed
- [ ] **Initializer/input separation** - Ensure initializers are not listed as graph inputs (ONNX IR v4+)

## Infrastructure

- [ ] **Fixed-point iteration** - Apply passes repeatedly until the graph stops changing (convergence)
- [ ] **`simplify` flag on `OnnxGraphBuilder`** - On by default
- [ ] **`simplify` flag on `ModelGen`** - On by default, forwarded to `OnnxGraphBuilder`
- [ ] **Disable simplification in tests** - `onnx-tests` and `onnx-ir` integration tests run without simplification

## Future / Nice-to-have

- [ ] **Large tensor threshold** - Skip folding operations that would produce tensors above a size limit
- [ ] **Subgraph simplification** - Simplify subgraphs inside control flow ops (If, Loop)
- [ ] **Operator-level skip list** - Allow skipping specific op types from simplification
