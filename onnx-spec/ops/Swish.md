# Swish

First introduced in opset **24**

## Description

Swish function takes one input data (Tensor<T>) and produces one output data (Tensor<T>) of the same shape,
where $Swish(x) = x * sigmoid(alpha * x)$.

## Attributes

- **alpha** (FLOAT, optional): Coefficient to multiply with input before sigmoid.

## Inputs (1 - 1)

- **X** (T): Input tensor

## Outputs (1 - 1)

- **Y** (T): Output tensor

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.
