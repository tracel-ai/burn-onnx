# Hardmax

First introduced in opset **1**

All versions: 1, 11, 13

## Description

The operator computes the hardmax values for the given input:

 Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

The "axis" attribute indicates the dimension along which Hardmax
will be performed. The output tensor has the same shape
and contains the Hardmax values of the corresponding input.

## Attributes

- **axis** (INT, optional): Describes the dimension Hardmax will be performed on.

## Inputs (1 - 1)

- **input** (T): The input tensor of rank >= axis.

## Outputs (1 - 1)

- **output** (T): The output values with the same shape as the input tensor.

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 11**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
