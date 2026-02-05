# MeanVarianceNormalization

First introduced in opset **9**

All versions: 9, 13

## Description

A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`

## Attributes

- **axes** (INTS, optional): A list of integers, along which to reduce. The default is to calculate along axes [0,2,3] for calculating mean and variance along each channel. Two variables with the same C-coordinate are associated with the same mean and variance.

## Inputs (1 - 1)

- **X** (T): Input tensor

## Outputs (1 - 1)

- **Y** (T): Output tensor

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to all numeric tensors.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 9**: Types: tensor(double), tensor(float), tensor(float16)
