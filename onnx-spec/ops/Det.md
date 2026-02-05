# Det

First introduced in opset **11**

All versions: 11, 22

## Description

Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).

## Inputs (1 - 1)

- **X** (T): Input tensor

## Outputs (1 - 1)

- **Y** (T): Output tensor

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to floating-point tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 11**: Types: tensor(double), tensor(float), tensor(float16)
