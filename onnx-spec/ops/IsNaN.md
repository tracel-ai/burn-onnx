# IsNaN

First introduced in opset **9**

All versions: 9, 13, 20

## Description

Returns which elements of the input are NaN.

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
- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 9**: Types: tensor(double), tensor(float), tensor(float16)
