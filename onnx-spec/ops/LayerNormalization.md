# LayerNormalization

First introduced in opset **17**

## Description

This is layer normalization defined in ONNX as function.
      The overall computation can be split into two stages.
      The first stage is standardization, which makes the
      normalized elements have zero mean and unit variances.
      The computation required by standardization can be
      described by the following equations.
      ```
      Mean = ReduceMean<axes=normalized_axes>(X)
      D = Sub(X, Mean)
      DD = Mul(D, D)
      Var = ReduceMean<axes=normalized_axes>(DD)
      VarEps = Add(Var, epsilon)
      StdDev = Sqrt(VarEps)
      InvStdDev = Reciprocal(StdDev)
      Normalized = Mul(D, InvStdDev)
      ```
      where `normalized_axes` is `[axis, ..., rank of X - 1]`.
      The variables `Var` and `StdDev` stand for variance and
      standard deviation, respectively. The second output is
      `Mean` and the last one is `InvStdDev`.
      Depending on `stash_type` attribute, the actual computation
      must happen in different floating-point precision.
      For example, if `stash_type` is 1, this operator casts
      all input variables to 32-bit float, perform the computation, and
      finally cast `Normalized` back to the original type of `X`.
      The second stage then scales and shifts the outcome of the
      first stage using
      ```
      NormalizedScaled = Mul(Normalized, Scale)
      Y = Add(NormalizedScaled, B)
      ```
      The second stage doesn't depends on `stash_type`.
      All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
      The same variable (i.e., input, output, and attribute) uses
      the same name in the equations above and this operator's definition.
      Let `d[i]` indicate the i-th dimension of `X`.
      If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
      the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
      `Y` and `X` have the same shape. This operator supports unidirectional broadcasting
      (tensors `Scale` and `B` should be unidirectional broadcastable to tensor `X`);
      for more details please check [the doc](Broadcasting.md).

## Attributes

- **axis** (INT, optional): The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). Negative value means counting dimensions from the back.
- **epsilon** (FLOAT, optional): The epsilon value to use to avoid division by zero.
- **stash_type** (INT, optional): Type of Mean and InvStdDev. This also specifies stage one's computation precision.

## Inputs (2 - 3)

- **X** (T): Tensor to be normalized.
- **Scale** (T): Scale tensor.
- **B** (T, optional): Bias tensor.

## Outputs (1 - 3)

- **Y** (T): Normalized tensor.
- **Mean** (U, optional): Saved mean used during training to speed up gradient computation
- **InvStdDev** (U, optional): Saved inverse standard deviation used during training to speed up gradient computation.

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input types and output Y type to float tensors.
- **U**: tensor(bfloat16), tensor(float)
  Type of Mean and InvStdDev tensors.
