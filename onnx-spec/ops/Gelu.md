# Gelu

First introduced in opset **20**

## Description

Gelu takes one input data (Tensor<T>) and produces one
output data (Tensor<T>) where the gaussian error linear units function,
$y = 0.5 * x * (1 + erf(x/sqrt(2)))$ is applied to the tensor elementwise.
If the attribute "approximate" is set to "tanh", the function estimation,
$y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))$ is used and applied
to the tensor elementwise.

## Attributes

- **approximate** (STRING, optional): Gelu approximation algorithm: `"tanh"`, `"none"`(default).`"none"`: do not use approximation.`"tanh"`: use tanh approximation.

## Inputs (1 - 1)

- **X** (T): Input tensor

## Outputs (1 - 1)

- **Y** (T): Output tensor

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.
