# SequenceEmpty

First introduced in opset **11**

## Description

Construct an empty tensor sequence, with given data type.

## Attributes

- **dtype** (INT, optional): (Optional) The data type of the tensors in the output sequence. The default type is 'float'.

## Outputs (1 - 1)

- **output** (S): Empty sequence.

## Type Constraints

- **S**: seq(tensor(bool)), seq(tensor(complex128)), seq(tensor(complex64)), seq(tensor(double)), seq(tensor(float)), seq(tensor(float16)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(int8)), seq(tensor(string)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(uint8))
  Constrain output types to any tensor type.
