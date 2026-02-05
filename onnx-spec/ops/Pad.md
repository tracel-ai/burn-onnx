# Pad

First introduced in opset **1**

All versions: 1, 2, 11, 13, 18, 19, 21, 23, 24

## Description

Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array

4) `wrap` - wrap-around padding as if the data tensor forms a torus


Example 1 (`constant` mode):

Insert 0 pads to the beginning of the second dimension.

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'constant'

constant_value = 0.0

output = [
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
```

Example 2 (`reflect` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'reflect'

output = [
    [1.0, 1.2, 1.0, 1.2],
    [2.3, 3.4, 2.3, 3.4],
    [4.5, 5.7, 4.5, 5.7],
]
```

Example 3 (`edge` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'edge'

output = [
    [1.0, 1.0, 1.0, 1.2],
    [2.3, 2.3, 2.3, 3.4],
    [4.5, 4.5, 4.5, 5.7],
]
```

Example 4 (`wrap` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [2, 1, 1, 1]

mode = 'wrap'

output = [
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
]
```

## Attributes

- **mode** (STRING, optional): Supported modes: `constant`(default), `reflect`, `edge`, `wrap`

## Inputs (2 - 4)

- **data** (T): Input tensor.
- **pads** (tensor(int64)): Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. `pads` should be a 1D tensor of shape [2 * num_axes] where `num_axes` refers to the number of elements in the `axes` input or the input rank if `axes` are not provided explicitly. `pads` format should be: [x1_begin, x2_begin, ..., x1_end, x2_end,...], where xi_begin is the number of pad values added at the beginning of axis `axes[i]` and xi_end, the number of pad values added at the end of axis `axes[i]`.
- **constant_value** (T, optional): (Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0, empty string or False).
- **axes** (Tind, optional): 1-D tensor of axes that `pads` apply to. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an axis is repeated. If not provided, all axes are assumed (`[0, 1, ..., input_rank-1]`).

## Outputs (1 - 1)

- **output** (T): Tensor after padding.

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
  Constrain input and output types to all tensor types up to IRv12.
- **Tind**: tensor(int32), tensor(int64)
  Constrain indices to integer types

## Version History

- **Opset 24**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 23**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 21**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
- **Opset 19**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 18**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 13**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 11**: Types: tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 2**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
