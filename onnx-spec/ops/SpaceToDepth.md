# SpaceToDepth

First introduced in opset **1**

All versions: 1, 13

## Description

SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.

## Attributes

- **blocksize** (INT, required): Blocks of [blocksize, blocksize] are moved.

## Inputs (1 - 1)

- **input** (T): Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.

## Outputs (1 - 1)

- **output** (T): Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all tensor types.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 1**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
