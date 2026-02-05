# Split

First introduced in opset **1**

All versions: 1, 2, 11, 13, 18

## Description

Split a tensor into a list of tensors, along the specified 'axis'.
Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
If the input 'split' is specified, it indicates the sizes of each output in the split.

## Attributes

- **axis** (INT, optional): Which axis to split on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] where r = rank(input).
- **num_outputs** (INT, optional): Number of outputs to split parts of the tensor into. If the tensor is not evenly splittable the last chunk will be smaller.

## Inputs (1 - 2)

- **input** (T): The tensor to split
- **split** (tensor(int64), optional): Optional length of each output. Values should be >= 0.Sum of the values must be equal to the dim value at 'axis' specified.

## Outputs (1 - 2147483647)

- **outputs** (T, variadic): One or more outputs forming list of tensors after splitting

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all tensor types.

## Version History

- **Opset 18**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 13**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 11**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 2**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
