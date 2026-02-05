# StringConcat

First introduced in opset **20**

## Description

StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support)

## Inputs (2 - 2)

- **X** (T): Tensor to prepend in concatenation
- **Y** (T): Tensor to append in concatenation

## Outputs (1 - 1)

- **Z** (T): Concatenated string tensor

## Type Constraints

- **T**: tensor(string)
  Inputs and outputs must be UTF-8 strings
