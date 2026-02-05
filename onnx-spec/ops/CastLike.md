# CastLike

First introduced in opset **15**

All versions: 15, 19, 21, 23, 24

## Description

The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.

## Attributes

- **round_mode** (STRING, optional): Rounding mode for conversion to float8e8m0. It only applies to casting to float8e8m0 and is `up` by default. `up`: round to nearest value away from zero, `down`: round to nearest value towards zero, `nearest`: round to nearest value and ties round up. Please refer to operator Cast description for further details.
- **saturate** (INT, optional): The parameter defines how the conversion behaves if an input value is out of range of the destination type. It only applies for float 8 conversion (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz, float8e8m0). It is true by default. Please refer to operator Cast description for further details.

## Inputs (2 - 2)

- **input** (T1): Input tensor to be cast.
- **target_type** (T2): The (first) input tensor will be cast to produce a tensor of the same type as this (second input) tensor.

## Outputs (1 - 1)

- **output** (T2): Output tensor produced by casting the first input tensor to have the same type as the second input tensor.

## Type Constraints

- **T1**: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
  Constrain input types. Casting from complex is not supported.
- **T2**: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
  Constrain output types. Casting to complex is not supported.

## Version History

- **Opset 24**: Types: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 23**: Types: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 21**: Types: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 19**: Types: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 15**: Types: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
