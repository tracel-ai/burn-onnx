# Elu

First introduced in opset **1**

All versions: 1, 6, 22

## Description

Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

## Attributes

- **alpha** (FLOAT, optional): Coefficient of ELU.

## Inputs (1 - 1)

- **X** (T): Input tensor

## Outputs (1 - 1)

- **Y** (T): Output tensor

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 6**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
