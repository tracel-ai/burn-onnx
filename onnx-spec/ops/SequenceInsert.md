# SequenceInsert

First introduced in opset **11**

## Description

Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.

## Inputs (2 - 3)

- **input_sequence** (S): Input sequence.
- **tensor** (T): Input tensor to be inserted into the input sequence.
- **position** (I, optional): Position in the sequence where the new tensor is inserted. It is optional and default is to insert to the back of the sequence. Negative value means counting positions from the back. Accepted range in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'. It is an error if any of the index values are out of bounds. It must be a scalar(tensor of empty shape).

## Outputs (1 - 1)

- **output_sequence** (S): Output sequence that contains the inserted tensor at given position.

## Type Constraints

- **T**: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain to any tensor type.
- **S**: seq(tensor(bool)), seq(tensor(complex128)), seq(tensor(complex64)), seq(tensor(double)), seq(tensor(float)), seq(tensor(float16)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(int8)), seq(tensor(string)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(uint8))
  Constrain to any tensor type.
- **I**: tensor(int32), tensor(int64)
  Constrain position to integral tensor. It must be a scalar(tensor of empty shape).
