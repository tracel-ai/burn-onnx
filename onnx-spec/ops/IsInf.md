# IsInf

First introduced in opset **10**

All versions: 10, 20

## Description

Map infinity to true and other values to false.

## Attributes

- **detect_negative** (INT, optional): (Optional) Whether map negative infinity to true. Default to 1 so that negative infinity induces true. Set this attribute to 0 if negative infinity should be mapped to false.
- **detect_positive** (INT, optional): (Optional) Whether map positive infinity to true. Default to 1 so that positive infinity induces true. Set this attribute to 0 if positive infinity should be mapped to false.

## Inputs (1 - 1)

- **X** (T1): input

## Outputs (1 - 1)

- **Y** (T2): output

## Type Constraints

- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  Constrain input types to float tensors.
- **T2**: tensor(bool)
  Constrain output types to boolean tensors.

## Version History

- **Opset 20**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
- **Opset 10**: Types: tensor(double), tensor(float)
