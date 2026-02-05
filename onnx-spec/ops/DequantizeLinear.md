# DequantizeLinear

First introduced in opset **10**

All versions: 10, 13, 19, 21, 23, 24

## Description

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.

`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 and 4-bit types quantization, but the dequantization formula remains the same
for consistency. The output type is determined by the attribute `output_dtype`. If `output_dtype` is not supplied then the output type
is the same as `x_scale`. The output type also determines the precision of the multiplication operation.

## Attributes

- **axis** (INT, optional): (Optional) The axis of the dequantizing dimension of the input tensor. Used for per-axis and blocked quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` where `r = rank(input)`.
- **block_size** (INT, optional): (Optional) The size of the quantization block (number of times every scale is replicated). Used only for blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, `y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is `[ceil(Di/Si), ceil(Di/(Si-1))-1]`
- **output_dtype** (INT, optional): (Optional) The output data type. If not supplied, the output data type is inferred from `x_scale` data type (`T2`)

## Inputs (2 - 3)

- **x** (T1): N-D quantized input tensor to be de-quantized.
- **x_scale** (T2): Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for per per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has the same shape as the input, except for one dimension in which blocking is performed.
- **x_zero_point** (T1, optional): Zero point for input `x`. Shape must match x_scale. It's optional. Zero point is 0 when it's not specified.

## Outputs (1 - 1)

- **y** (T3): N-D full precision output tensor. It has the same shape as input `x`. The data type is specified by the `output_dtype` attribute or, in its absence, the type of `x_scale`.

## Type Constraints

- **T1**: tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int8), tensor(uint16), tensor(uint4), tensor(uint8)
  The type of the inputs 'x_zero_point' and 'x'.
- **T2**: tensor(bfloat16), tensor(float), tensor(float16), tensor(float8e8m0)
  The type of the input 'x_scale'.
- **T3**: tensor(bfloat16), tensor(float), tensor(float16)
  The type of the output 'y'.

## Version History

- **Opset 24**: Types: tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int8), tensor(uint16), tensor(uint4), tensor(uint8)
- **Opset 23**: Types: tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int8), tensor(uint16), tensor(uint4), tensor(uint8)
- **Opset 21**: Types: tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int8), tensor(uint16), tensor(uint4), tensor(uint8)
- **Opset 19**: Types: tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int32), tensor(int8), tensor(uint8)
- **Opset 13**: Types: tensor(int32), tensor(int8), tensor(uint8)
- **Opset 10**: Types: tensor(int32), tensor(int8), tensor(uint8)
