# QLinearMatMul

First introduced in opset **10**

All versions: 10, 21

## Description

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.

## Inputs (8 - 8)

- **a** (T1): N-dimensional quantized matrix a
- **a_scale** (TS): scale of quantized input a
- **a_zero_point** (T1): zero point of quantized input a
- **b** (T2): N-dimensional quantized matrix b
- **b_scale** (TS): scale of quantized input b
- **b_zero_point** (T2): zero point of quantized input b
- **y_scale** (TS): scale of quantized output y
- **y_zero_point** (T3): zero point of quantized output y

## Outputs (1 - 1)

- **y** (T3): Quantized matrix multiply results from a * b

## Type Constraints

- **TS**: tensor(bfloat16), tensor(float), tensor(float16)
  Constrain scales.
- **T1**: tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int8), tensor(uint8)
  The type of input a and its zeropoint.
- **T2**: tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int8), tensor(uint8)
  The type of input b and its zeropoint.
- **T3**: tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int8), tensor(uint8)
  The type of the output and its zeropoint.

## Version History

- **Opset 21**: Types: tensor(bfloat16), tensor(float), tensor(float16)
- **Opset 10**: Types: tensor(int8), tensor(uint8)
