# LogSoftmax

First introduced in opset **1**

All versions: 1, 11, 13

## Description

The operator computes the log of softmax values for the given input:

 LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

The "axis" attribute indicates the dimension along which LogSoftmax
will be performed. The output tensor has the same shape
and contains the LogSoftmax values of the corresponding input.

## Attributes

- **axis** (INT, optional): Describes the dimension LogSoftmax will be performed on.

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
