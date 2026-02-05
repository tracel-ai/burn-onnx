# InstanceNormalization

First introduced in opset **1**

All versions: 1, 6, 22

## Description

Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

## Attributes

- **epsilon** (FLOAT, optional): The epsilon value to use to avoid division by zero.

## Inputs (3 - 3)

- **input** (T): Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
- **scale** (T): The input 1-dimensional scale tensor of size C.
- **B** (T): The input 1-dimensional bias tensor of size C.

## Outputs (1 - 1)

- **output** (T): The output tensor of the same shape as input.

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 6**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
