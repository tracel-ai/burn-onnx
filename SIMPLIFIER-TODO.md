# ONNX Simplifier - Feature Tracking

Tracking issue: [#61](https://github.com/tracel-ai/burn-onnx/issues/61)

Reference: [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

## Design

- Implement as a `simplify` module within `onnx-ir` (not a separate crate)
- Add `simplify` flag to `OnnxGraphBuilder` (default: `true`)
- Expose `simplify` flag in `ModelGen` (default: `true`)
- Disable simplification in `onnx-tests` and `onnx-ir` integration tests

## Infrastructure

- [x] `simplify` flag on `OnnxGraphBuilder` (default: true)
- [x] `simplify` flag on `ModelGen` (default: true, forwarded to `OnnxGraphBuilder`)
- [x] Disable simplification in `onnx-tests` and `onnx-ir` integration tests
- [x] Fixed-point iteration: apply passes repeatedly until the graph stops changing

## Elimination Passes

Remove unnecessary nodes from the graph.

- [x] Dead node elimination: remove nodes whose outputs are not consumed by any other node
      or graph output
- [x] Redundant node elimination: merge duplicate nodes that have identical op type, attributes,
      and inputs (CSE - common subexpression elimination)
- [x] Identity elimination (in post-processing phase via is_noop)
- [x] No-op Cast elimination: remove Cast where input dtype == output dtype
- [x] No-op Transpose elimination: remove Transpose with identity permutation [0,1,2,...]
- [x] No-op Reshape elimination: remove Reshape where output shape == input shape
- [x] No-op Pad elimination: remove Pad where all pad values are 0
- [x] No-op Dropout elimination: remove Dropout in inference mode (always no-op, inference only)
- [x] No-op Expand elimination: remove Expand where output shape == input shape
- [x] No-op Concat elimination: remove Concat with single input (replace with Identity)
- [x] No-op Split elimination: remove Split with single output
- [x] No-op Flatten elimination: remove Flatten where input is already rank 2
- [x] Consecutive idempotent op elimination: `Relu(Relu(x))` -> `Relu(x)`, same for
      Ceil, Floor, Round, Sign, Abs
- [x] Identity element elimination: `x + 0 = x`, `0 + x = x`, `x - 0 = x`,
      `x * 1 = x`, `1 * x = x`, `x / 1 = x`, `x ** 1 = x`

## Pattern-Based Simplifications

Replace common multi-node patterns with simpler equivalents.

- [x] Shape+Gather+Unsqueeze+Concat+Reshape -> Transpose/Reshape: detect patterns where
      individual shape dimensions are gathered, unsqueezed, concatenated, and used in Reshape
      to effectively implement a permutation (see diagram in issue). Replace with a single
      Transpose or simplified Reshape
- [x] Constant Shape propagation: when a tensor's shape is statically known, replace
      Shape -> Gather chains with constant values
- [x] Shape op elimination: when input has fully static shape, replace entire Shape node
      with a constant tensor (generalizes the Shape->Gather pass above)
- [x] Slice-after-Shape elimination: `Shape -> Slice` with known dims -> constant

### Deferred (requires subgraph handling)

- [ ] Constant If elimination: when the condition of an If node is a constant, replace with
      the taken branch's subgraph
- [ ] Constant Loop elimination: when trip count is a constant and condition is always true,
      consider unrolling or simplifying

## Constant Folding (future work)

Evaluate nodes where all inputs are compile-time constants and replace with constant tensors.
Unlike onnx-simplifier (which uses ONNX Runtime), we evaluate ops directly on constant data.

**Approach**: Build a core framework that detects nodes with all-static inputs, evaluates them
using burn's CPU backend (or manual evaluation on raw bytes), and replaces with constant tensors.
Roll out incrementally per op category.

**Priority order** (based on frequency in real models):
1. Core framework: detect all-constant-input nodes, evaluate, replace with constants
2. Arithmetic: Add, Sub, Mul, Div, Pow, Neg, Abs, Sqrt, Reciprocal, Exp, Log, Ceil, Floor, Mod
3. Tensor manipulation: Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Expand, Slice, Concat,
   Split, Tile
4. Cast on constants
5. Comparison/logical: Equal, Greater, Less, GreaterOrEqual, LessOrEqual, Not, And, Or, Xor
6. Shape ops: Shape, Size, Gather, GatherElements, ScatterElements
7. Reductions: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd
8. Where/Select on constants
9. MatMul/Gemm on small constant matrices

**Safeguards**:
- Large tensor threshold: skip folding ops that produce tensors above a configurable size
  (Tile, ConstantOfShape, Expand are common culprits)
- Skip non-deterministic ops: RandomUniform, RandomNormal, RandomUniformLike,
  RandomNormalLike, Multinomial

**Note**: Many high-value constant expression patterns (shape computation subgraphs) are already
handled by the pattern-based simplification passes above. Constant folding would cover the
remaining cases where arbitrary constant expressions appear in models.

## Fusion Passes (not planned)

Op fusion is handled at runtime by Burn's backend fusion system, not at the ONNX import level.
These are listed here for reference only.

- BatchNorm into Conv: fold BN parameters into Conv weights/bias
- Pad into Conv/Pool: merge explicit Pad into padding attributes
- Add bias into Conv: fold constant Add into Conv's bias parameter
- Consecutive Squeeze/Transpose/Reshape fusion: merge chained ops into one
- Transpose into Gemm: absorb into transA/transB attributes
