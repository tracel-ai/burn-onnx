# Shrink

First introduced in opset **9**

## Description

Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.

## Attributes

- **bias** (FLOAT, optional): The bias value added to output. Default is 0.
- **lambd** (FLOAT, optional): The lambd value for the Shrink formulation. Default is 0.5.

## Inputs (1 - 1)

- **input** (T): The input data as Tensor.

## Outputs (1 - 1)

- **output** (T): The output.

## Type Constraints

- **T**: tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input to only numeric types.
