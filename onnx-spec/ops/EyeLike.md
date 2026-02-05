# EyeLike

First introduced in opset **9**

All versions: 9, 22

## Description

Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

## Attributes

- **dtype** (INT, optional): (Optional) The data type for the elements of the output tensor. If not specified, the data type of the input tensor T1 is used.
- **k** (INT, optional): (Optional) Index of the diagonal to be populated with ones. Default is 0. If T2 is the output, this op sets T2[i, i+k] = 1. k = 0 populates the main diagonal, k > 0 populates an upper diagonal,  and k < 0 populates a lower diagonal.

## Inputs (1 - 1)

- **input** (T1): 2D input tensor to copy shape, and optionally, type information from.

## Outputs (1 - 1)

- **output** (T2): Output tensor, same shape as input tensor T1.

## Type Constraints

- **T1**: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input types. Strings and complex are not supported.
- **T2**: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain output types. Strings and complex are not supported.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 9**: Types: tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
