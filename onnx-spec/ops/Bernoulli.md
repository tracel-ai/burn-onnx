# Bernoulli

First introduced in opset **15**

All versions: 15, 22

## Description

Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).

## Attributes

- **dtype** (INT, optional): The data type for the elements of the output tensor. if not specified, we will use the data type of the input tensor.
- **seed** (FLOAT, optional): (Optional) Seed to the random generator, if not specified we will auto generate one.

## Inputs (1 - 1)

- **input** (T1): All values in input have to be in the range:[0, 1].

## Outputs (1 - 1)

- **output** (T2): The returned output tensor only has values 0 or 1, same shape as input tensor.

## Type Constraints

- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input types to float tensors.
- **T2**: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain output types to all numeric tensors and bool tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 15**: Types: tensor(double), tensor(float), tensor(float16)
