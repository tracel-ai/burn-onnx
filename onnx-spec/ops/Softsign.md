# Softsign

First introduced in opset **1**

All versions: 1, 22

## Description

Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.

## Inputs (1 - 1)

- **input** (T): Input tensor

## Outputs (1 - 1)

- **output** (T): The softsign (x/(1+|x|)) values of the input tensor computed element-wise

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
