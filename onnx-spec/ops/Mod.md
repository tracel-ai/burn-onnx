# Mod

First introduced in opset **10**

All versions: 10, 13

## Description

Performs an element-wise binary modulo operation.
The semantics and supported data types depend on the value of the `fmod` attribute which must be `0` (default), or `1`.

If the `fmod` attribute is set to `0`, `T` is constrained to integer data types and the semantics follow that of the Python `%`-operator.
The sign of the result is that of the divisor.

If `fmod` is set to `1`, the behavior of this operator follows that of the `fmod` function in C and `T` is constrained to floating point data types.
The result of this operator is the remainder of the division operation `x / y` where `x` and `y` are respective elements of `A` and `B`. The result is exactly the value `x - n * y`, where `n` is `x / y` with its fractional part truncated.
The returned value has the same sign as `x` (except if `x` is `-0`) and is less or equal to `|y|` in magnitude.
The following special cases apply when `fmod` is set to `1`:
- If `x` is `-0` and `y` is greater than zero, either `+0` or `-0` may be returned.
- If `x` is `±∞` and `y` is not `NaN`, `NaN` is returned.
- If `y` is `±0` and `x` is not `NaN`, `NaN` should be returned.
- If `y` is `±∞` and `x` is finite, `x` is returned.
- If either argument is `NaN`, `NaN` is returned.

This operator supports **multidirectional (i.e., NumPy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

## Attributes

- **fmod** (INT, optional): Whether the operator should behave like fmod (default=0 meaning it will do integer mods); Set this to 1 to force fmod treatment

## Inputs (2 - 2)

- **A** (T): Dividend tensor
- **B** (T): Divisor tensor

## Outputs (1 - 1)

- **C** (T): Remainder tensor

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to high-precision numeric tensors.

## Version History

- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 10**: Types: tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
