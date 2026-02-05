# SequenceLength

First introduced in opset **11**

## Description

Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.

## Inputs (1 - 1)

- **input_sequence** (S): Input sequence.

## Outputs (1 - 1)

- **length** (I): Length of input sequence. It must be a scalar(tensor of empty shape).

## Type Constraints

- **S**: seq(tensor(bool)), seq(tensor(complex128)), seq(tensor(complex64)), seq(tensor(double)), seq(tensor(float)), seq(tensor(float16)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(int8)), seq(tensor(string)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(uint8))
  Constrain to any tensor type.
- **I**: tensor(int64)
  Constrain output to integral tensor. It must be a scalar(tensor of empty shape).
