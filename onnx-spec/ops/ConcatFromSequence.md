# ConcatFromSequence

First introduced in opset **11**

## Description

Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.

## Attributes

- **axis** (INT, required): Which axis to concat on. Accepted range in `[-r, r - 1]`, where `r` is the rank of input tensors. When `new_axis` is 1, accepted range is `[-r - 1, r]`.
- **new_axis** (INT, optional): Insert and concatenate on a new axis or not, default 0 means do not insert new axis.

## Inputs (1 - 1)

- **input_sequence** (S): Sequence of tensors for concatenation

## Outputs (1 - 1)

- **concat_result** (T): Concatenated tensor

## Type Constraints

- **S**: seq(tensor(bool)), seq(tensor(complex128)), seq(tensor(complex64)), seq(tensor(double)), seq(tensor(float)), seq(tensor(float16)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(int8)), seq(tensor(string)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(uint8))
  Constrain input types to any tensor type.
- **T**: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain output types to any tensor type.
