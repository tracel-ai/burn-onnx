# LpNormalization

First introduced in opset **1**

All versions: 1, 22

## Description

Given a matrix, apply Lp-normalization along the provided axis.

## Attributes

- **axis** (INT, optional): The axis on which to apply normalization, -1 mean last axis.
- **p** (INT, optional): The order of the normalization, only 1 or 2 are supported.

## Inputs (1 - 1)

- **input** (T): Input matrix

## Outputs (1 - 1)

- **output** (T): Matrix after normalization

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
