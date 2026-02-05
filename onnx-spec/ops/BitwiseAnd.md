# BitwiseAnd

First introduced in opset **18**

## Description

Returns the tensor resulting from performing the bitwise `and` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

## Inputs (2 - 2)

- **A** (T): First input operand for the bitwise operator.
- **B** (T): Second input operand for the bitwise operator.

## Outputs (1 - 1)

- **C** (T): Result tensor.

## Type Constraints

- **T**: tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input to integer tensors.
