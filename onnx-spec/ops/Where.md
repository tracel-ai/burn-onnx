# Where

First introduced in opset **9**

All versions: 9, 16

## Description

Return elements, either from X or Y, depending on condition.
Where behaves like
[numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

## Inputs (3 - 3)

- **condition** (B): When True (nonzero), yield X, otherwise yield Y
- **X** (T): values selected at indices where condition is True
- **Y** (T): values selected at indices where condition is False

## Outputs (1 - 1)

- **output** (T): Tensor of shape equal to the broadcasted shape of condition, X, and Y.

## Type Constraints

- **B**: tensor(bool)
  Constrain to boolean tensors.
- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all tensor types (including bfloat).

## Version History

- **Opset 16**: Types: tensor(bool)
- **Opset 9**: Types: tensor(bool)
