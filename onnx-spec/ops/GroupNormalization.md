# GroupNormalization

First introduced in opset **18**

All versions: 18, 21

## Description

A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each channel. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

The overall computation has two stages: the first stage normalizes the elements to
have zero mean and unit variance for each instance in each group, and the second
stage scales and shifts the results of the first stage. The floating-point precision
used in the first stage is determined by the `stash_type` attribute. For example,
if `stash_type` is 1, the operator casts all input variables to 32-bit float,
performs the computation, and finally casts the normalized results back to the
original type of `X`. The second stage does not depend on `stash_type`.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.

## Attributes

- **epsilon** (FLOAT, optional): The epsilon value to use to avoid division by zero.
- **num_groups** (INT, required): The number of groups of channels. It should be a divisor of the number of channels `C`.
- **stash_type** (INT, optional): The floating-point precision used in stage one of the computation.

## Inputs (3 - 3)

- **X** (T): Input data tensor. Dimensions for image cases are `(N x C x H x W)`, where `N` is the batch size, `C` is the number of channels, and `H` and `W` are the height and width of the data. Statistics are computed for every group of channels over `C`, `H`, and `W`. For non-image cases, the dimensions are in the form of `(N x C x D1 x D2 ... Dn)`.
- **scale** (T): Scale tensor of shape `(C)`.
- **bias** (T): Bias tensor of shape `(C)`.

## Outputs (1 - 1)

- **Y** (T): The output tensor of the same shape as `X`.

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.

## Version History

- **Opset 21**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 18**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
