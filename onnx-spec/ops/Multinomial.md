# Multinomial

First introduced in opset **7**

All versions: 7, 22

## Description

Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.

## Attributes

- **dtype** (INT, optional): (Optional) The data type for the elements of the output tensor, if not specified, we will use int32.
- **sample_size** (INT, optional): Number of times to sample.
- **seed** (FLOAT, optional): (Optional) Seed to the random generator, if not specified we will auto generate one.

## Inputs (1 - 1)

- **input** (T1): Input tensor with shape [batch_size, class_size], where class_size is the number of all possible outcomes. Each value along the axis zero represents the unnormalized log-probability of each corresponding outcome in a batch.

## Outputs (1 - 1)

- **output** (T2): Output tensor with shape [batch_size, sample_size], where sample_size is the number of times to sample. Each value along the axis zero represents the outcome of the corresponding sample in a batch.

## Type Constraints

- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input types to float tensors.
- **T2**: tensor(int32), tensor(int64)
  Constrain output types to integral tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 7**: Types: tensor(double), tensor(float), tensor(float16)
