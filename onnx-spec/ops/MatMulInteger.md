# MatMulInteger

First introduced in opset **10**

## Description

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

## Inputs (2 - 4)

- **A** (T1): N-dimensional matrix A
- **B** (T2): N-dimensional matrix B
- **a_zero_point** (T1, optional): Zero point tensor for input 'A'. It's optional and default value is 0. It could be a scalar or N-D tensor. Scalar refers to per tensor quantization whereas N-D refers to per row quantization. If the input is 2D of shape [M, K] then zero point tensor may be an M element vector [zp_1, zp_2, ..., zp_M]. If the input is N-D tensor with shape [D1, D2, M, K] then zero point tensor may have shape [D1, D2, M, 1].
- **b_zero_point** (T2, optional): Zero point tensor for input 'B'. It's optional and default value is 0. It could be a scalar or a N-D tensor, Scalar refers to per tensor quantization whereas N-D refers to per col quantization. If the input is 2D of shape [K, N] then zero point tensor may be an N element vector [zp_1, zp_2, ..., zp_N]. If the input is N-D tensor with shape [D1, D2, K, N] then zero point tensor may have shape [D1, D2, 1, N].

## Outputs (1 - 1)

- **Y** (T3): Matrix multiply results from A * B

## Type Constraints

- **T1**: tensor(int8), tensor(uint8)
  Constrain input A data type to 8-bit integer tensor.
- **T2**: tensor(int8), tensor(uint8)
  Constrain input B data type to 8-bit integer tensor.
- **T3**: tensor(int32)
  Constrain output Y data type as 32-bit integer tensor.
