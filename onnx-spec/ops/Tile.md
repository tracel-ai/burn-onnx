# Tile

First introduced in opset **1**

All versions: 1, 6, 13

## Description

Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

## Inputs (2 - 2)

- **input** (T): Input tensor of any shape.
- **repeats** (T1): 1D int64 tensor of the same length as input's dimension number, includes numbers of repeated copies along input's dimensions.

## Outputs (1 - 1)

- **output** (T): Output tensor of the same dimensions and type as tensor input. output_dim[i] = input_dim[i] * repeats[i]

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all tensor types.
- **T1**: tensor(int64)
  Constrain repeat's type to int64 tensors.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 6**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
