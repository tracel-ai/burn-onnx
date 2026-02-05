# MatMul

First introduced in opset **1**

All versions: 1, 9, 13

## Description

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).

## Inputs (2 - 2)

- **A** (T): N-dimensional matrix A
- **B** (T): N-dimensional matrix B

## Outputs (1 - 1)

- **Y** (T): Matrix multiply results from A * B

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
  Constrain input and output types to float/int tensors.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 9**: Types: tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
