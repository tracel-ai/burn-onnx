# Pow

First introduced in opset **1**

All versions: 1, 7, 12, 13, 15

## Description

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

## Inputs (2 - 2)

- **X** (T): First operand, base of the exponent.
- **Y** (T1): Second operand, power of the exponent.

## Outputs (1 - 1)

- **Z** (T): Output tensor

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64)
  Constrain input X and output types to float/int tensors.
- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input Y types to float/int tensors.

## Version History

- **Opset 15**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64)
- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64)
- **Opset 12**: Types: tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64)
- **Opset 7**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
