# Selu

First introduced in opset **1**

All versions: 1, 6, 22

## Description

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.

## Attributes

- **alpha** (FLOAT, optional): Coefficient of SELU default to 1.67326319217681884765625 (i.e., float32 approximation of 1.6732632423543772848170429916717).
- **gamma** (FLOAT, optional): Coefficient of SELU default to 1.05070102214813232421875 (i.e., float32 approximation of 1.0507009873554804934193349852946).

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
