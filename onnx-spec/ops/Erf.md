# Erf

First introduced in opset **9**

All versions: 9, 13

## Description

Computes the error function of the given input tensor element-wise.

## Inputs (1 - 1)

- **input** (T): Input tensor

## Outputs (1 - 1)

- **output** (T): The error function of the input tensor computed element-wise. It has the same shape and type of the input.

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all numeric tensors.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 9**: Types: tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
