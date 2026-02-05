# SequenceErase

First introduced in opset **11**

## Description

Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.

## Inputs (1 - 2)

- **input_sequence** (S): Input sequence.
- **position** (I, optional): Position of the tensor in the sequence. Negative value means counting positions from the back. Accepted range in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'. It is an error if any of the index values are out of bounds. It must be a scalar(tensor of empty shape).

## Outputs (1 - 1)

- **output_sequence** (S): Output sequence that has the tensor at the specified position removed.

## Type Constraints

- **S**: seq(tensor(bool)), seq(tensor(complex128)), seq(tensor(complex64)), seq(tensor(double)), seq(tensor(float)), seq(tensor(float16)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(int8)), seq(tensor(string)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(uint8))
  Constrain to any tensor type.
- **I**: tensor(int32), tensor(int64)
  Constrain position to integral tensor. It must be a scalar(tensor of empty shape).
