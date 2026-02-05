# RandomNormalLike

First introduced in opset **1**

All versions: 1, 22

## Description

Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.

## Attributes

- **dtype** (INT, optional): (Optional) The data type for the elements of the output tensor, if not specified, we will use the data type of the input tensor.
- **mean** (FLOAT, optional): The mean of the normal distribution.
- **scale** (FLOAT, optional): The standard deviation of the normal distribution.
- **seed** (FLOAT, optional): (Optional) Seed to the random generator, if not specified we will auto generate one.

## Inputs (1 - 1)

- **input** (T1): Input tensor to copy shape and optionally type information from.

## Outputs (1 - 1)

- **output** (T2): Output tensor of random values drawn from normal distribution

## Type Constraints

- **T1**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.
- **T2**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain output types to float tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 1**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
