# And

First introduced in opset **1**

All versions: 1, 7

## Description

Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

## Inputs (2 - 2)

- **A** (T): First input operand for the logical operator.
- **B** (T): Second input operand for the logical operator.

## Outputs (1 - 1)

- **C** (T1): Result tensor.

## Type Constraints

- **T**: tensor(bool)
  Constrain input to boolean tensor.
- **T1**: tensor(bool)
  Constrain output to boolean tensor.

## Version History

- **Opset 7**: Types: tensor(bool)
- **Opset 1**: Types: tensor(bool)
