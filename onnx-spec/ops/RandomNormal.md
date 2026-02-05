# RandomNormal

First introduced in opset **1**

All versions: 1, 22

## Description

Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.

## Attributes

- **dtype** (INT, optional): The data type for the elements of the output tensor. Default is TensorProto::FLOAT.
- **mean** (FLOAT, optional): The mean of the normal distribution.
- **scale** (FLOAT, optional): The standard deviation of the normal distribution.
- **seed** (FLOAT, optional): (Optional) Seed to the random generator, if not specified we will auto generate one.
- **shape** (INTS, required): The shape of the output tensor.

## Outputs (1 - 1)

- **output** (T): Output tensor of random values drawn from normal distribution

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain output types to float tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
