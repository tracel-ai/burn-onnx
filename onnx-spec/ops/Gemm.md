# Gemm

First introduced in opset **1**

All versions: 1, 6, 7, 9, 11, 13

## Description

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

* A' = transpose(A) if transA else A
* B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

## Attributes

- **alpha** (FLOAT, optional): Scalar multiplier for the product of input tensors A * B.
- **beta** (FLOAT, optional): Scalar multiplier for input tensor C.
- **transA** (INT, optional): Whether A should be transposed
- **transB** (INT, optional): Whether B should be transposed

## Inputs (2 - 3)

- **A** (T): Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
- **B** (T): Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
- **C** (T, optional): Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).

## Outputs (1 - 1)

- **Y** (T): Output tensor of shape (M, N).

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
  Constrain input and output types to float/int tensors.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 11**: Types: tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 9**: Types: tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 7**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 6**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
