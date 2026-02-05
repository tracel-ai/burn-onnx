# Upsample

First introduced in opset **1**

All versions: 1, 7, 9, 10

## Description

Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).

## Attributes

- **mode** (STRING, optional): Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)

## Inputs (2 - 2)

- **X** (T): N-D tensor
- **scales** (tensor(float)): The scale array along each dimension. It takes value greater than or equal to 1. The number of elements of 'scales' should be the same as the rank of input 'X'.

## Outputs (1 - 1)

- **Y** (T): N-D tensor after resizing

## Type Constraints

- **T**: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input 'X' and output 'Y' to all tensor types.

## Version History

- **Opset 10**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 9**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 7**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 1**: Types: tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64)
