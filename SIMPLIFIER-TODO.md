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
- [ ] Fixed-point iteration: apply passes repeatedly until the graph stops changing

## Constant Folding

DO NOT WORK ON CONSTANT FOLDING!!

Evaluate nodes where all inputs are compile-time constants and replace with constant tensors.
Unlike onnx-simplifier (which uses ONNX Runtime), we evaluate ops directly on constant data.

- [ ] Core framework: detect nodes with all-static inputs, evaluate, replace with constants
- [ ] Arithmetic ops on constants: Add, Sub, Mul, Div, Pow, Mod, Sqrt, Reciprocal, Neg, Abs,
      Ceil, Floor, Exp, Log
- [ ] Comparison ops on constants: Equal, Greater, Less, GreaterOrEqual, LessOrEqual, Not, And,
      Or, Xor
- [ ] Tensor manipulation on constants: Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Expand,
      Slice, Concat, Split, Tile
- [ ] Reduction ops on constants: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd
- [ ] Shape ops: Shape, Size, Gather (on constants), GatherElements, ScatterElements
- [ ] Cast on constants
- [ ] Where/Select on constants
- [ ] MatMul/Gemm on small constant matrices
- [ ] Large tensor threshold: skip folding ops that produce tensors above a configurable size
      (Tile, ConstantOfShape, Expand are common culprits)
- [ ] Skip non-deterministic ops: RandomUniform, RandomNormal, RandomUniformLike,
      RandomNormalLike, Multinomial

## Elimination Passes

Remove unnecessary nodes from the graph.

I THINK MOST OF THEM ARE ALREADY DONE IN POST-PROCESSING PHASE

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
- [ ] Unused initializer elimination: remove initializers/constants not consumed by any node

## Fusion Passes

DO NOT WORK ON FUSION PASS

Combine multiple nodes into a single more efficient operation.

- [ ] BatchNorm into Conv: fold BN parameters (scale, bias, mean, var) into Conv weights/bias
- [ ] Pad into Conv/Pool: merge explicit Pad node into the padding attribute of Conv/Pool ops
- [ ] Add bias into Conv: fold a following Add (constant bias) into Conv's bias parameter
- [ ] Consecutive Squeeze fusion: merge chained Squeeze ops into one
- [ ] Consecutive Transpose fusion: merge chained Transpose ops into one (compose permutations)
- [ ] Transpose into Gemm: absorb Transpose into Gemm's transA/transB attributes
- [ ] Consecutive Reshape fusion: merge chained Reshape ops into one

## Pattern-Based Simplifications

Replace common multi-node patterns with simpler equivalents.

- [x] Shape+Gather+Unsqueeze+Concat+Reshape -> Transpose/Reshape: detect patterns where
      individual shape dimensions are gathered, unsqueezed, concatenated, and used in Reshape
      to effectively implement a permutation (see diagram in issue). Replace with a single
      Transpose or simplified Reshape
- [x] Constant Shape propagation: when a tensor's shape is statically known, replace
      Shape -> Gather chains with constant values
- [ ] Shape op elimination: when input has fully static shape, replace entire Shape node
      with a constant tensor (generalizes the Shape->Gather pass above)
- [ ] Slice-after-Shape elimination: `Shape -> Slice` with known dims -> constant

### Deferred (requires subgraph handling)

- [ ] Constant If elimination: when the condition of an If node is a constant, replace with
      the taken branch's subgraph
- [ ] Constant Loop elimination: when trip count is a constant and condition is always true,
      consider unrolling or simplifying

## Shape Inference Propagation

DO NOT WORK ON THIS
