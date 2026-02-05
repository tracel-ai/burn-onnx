# Exp

First introduced in opset **1**

All versions: 1, 6, 13

## Description

Calculates the exponential of the given input tensor, element-wise.

## Inputs (1 - 1)

- **input** (T): Input tensor

## Outputs (1 - 1)

- **output** (T): The exponential of the input tensor computed element-wise

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 6**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
