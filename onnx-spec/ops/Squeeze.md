# Squeeze

First introduced in opset **1**

All versions: 1, 11, 13, 21, 23, 24

## Description

Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.

## Inputs (1 - 2)

- **data** (T): Tensors with at least max(dims) dimensions.
- **axes** (tensor(int64), optional): 1D tensor of integers indicating the dimensions to squeeze. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

## Outputs (1 - 1)

- **squeezed** (T): Reshaped tensor with same data as input.

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
  Constrain input and output types to all tensor types up to IRv12.

## Version History

- **Opset 24**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 23**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 21**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 13**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 11**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 1**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
