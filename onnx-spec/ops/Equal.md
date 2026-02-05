# Equal

First introduced in opset **1**

All versions: 1, 7, 11, 13, 19

## Description

Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

## Inputs (2 - 2)

- **A** (T): First input operand for the logical operator.
- **B** (T): Second input operand for the logical operator.

## Outputs (1 - 1)

- **C** (T1): Result tensor.

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input types to all (non-complex) tensors.
- **T1**: tensor(bool)
  Constrain output to boolean tensor.

## Version History

- **Opset 19**: Types: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 13**: Types: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 11**: Types: tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 7**: Types: tensor(bool), tensor(int32), tensor(int64)
- **Opset 1**: Types: tensor(bool), tensor(int32), tensor(int64)
