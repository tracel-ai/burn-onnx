# Dropout

First introduced in opset **1**

All versions: 1, 6, 7, 10, 12, 13, 22

## Description

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

## Attributes

- **seed** (INT, optional): (Optional) Seed to the random generator, if not specified we will auto generate one.

## Inputs (1 - 3)

- **data** (T): The input data as Tensor.
- **ratio** (T1, optional): The ratio of random dropout, with value in [0, 1). If set to 0, the output would be a simple copy of the input. If it's non-zero, output will be a random dropout of the scaled input, which is typically the case during training. It is an optional value, if not specified it will default to 0.5.
- **training_mode** (T2, optional): If set to true then it indicates dropout is being used for training. It is an optional value hence unless specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where nothing will be dropped from the input data and if mask is requested as output it will contain all ones.

## Outputs (1 - 2)

- **output** (T): The output.
- **mask** (T2, optional): The output mask.

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  Constrain input and output types to float tensors.
- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
  Constrain input 'ratio' types to float tensors.
- **T2**: tensor(bool)
  Constrain output 'mask' types to boolean tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 12**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 10**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 7**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 6**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
