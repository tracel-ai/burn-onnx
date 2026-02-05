# ConstantOfShape

First introduced in opset **9**

All versions: 9, 20, 21, 23, 24

## Description

Generate a tensor with given value and shape.

## Attributes

- **value** (TENSOR, optional): (Optional) The value of the output elements.Should be a one-element tensor. If not specified, it defaults to a tensor of value 0 and datatype float32

## Inputs (1 - 1)

- **input** (T1): 1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar. All values must be >= 0.

## Outputs (1 - 1)

- **output** (T2): Output tensor of shape specified by 'input'.If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype defaults to float32.

## Type Constraints

- **T1**: tensor(int64)
  Constrain input types.
- **T2**: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
  Constrain output types to be numerics or boolean.

## Version History

- **Opset 24**: Types: tensor(int64)
- **Opset 23**: Types: tensor(int64)
- **Opset 21**: Types: tensor(int64)
- **Opset 20**: Types: tensor(int64)
- **Opset 9**: Types: tensor(int64)
