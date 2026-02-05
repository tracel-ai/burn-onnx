# RandomUniform

First introduced in opset **1**

All versions: 1, 22

## Description

Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.

## Attributes

- **dtype** (INT, optional): The data type for the elements of the output tensor. If not specified, default is TensorProto::FLOAT.
- **high** (FLOAT, optional): Upper boundary of the output values.
- **low** (FLOAT, optional): Lower boundary of the output values.
- **seed** (FLOAT, optional): (Optional) Seed to the random generator, if not specified we will auto generate one.
- **shape** (INTS, required): The shape of the output tensor.

## Outputs (1 - 1)

- **output** (T): Output tensor of random values drawn from uniform distribution

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain output types to float tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
