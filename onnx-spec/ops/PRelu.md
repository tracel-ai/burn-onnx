# PRelu

First introduced in opset **1**

All versions: 1, 6, 7, 9, 16

## Description

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).

## Inputs (2 - 2)

- **X** (T): Input tensor
- **slope** (T): Slope tensor. The shape of slope can be smaller than first input X; if so, its shape must be unidirectional broadcastable to X

## Outputs (1 - 1)

- **Y** (T): Output tensor (same size as X)

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
  Constrain input and output types to float/int tensors.

## Version History

- **Opset 16**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 9**: Types: tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 7**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 6**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
