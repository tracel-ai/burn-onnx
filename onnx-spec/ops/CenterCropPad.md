# CenterCropPad

First introduced in opset **18**

## Description

Center crop or pad an input to given dimensions.

The crop/pad dimensions can be specified for a subset of the `axes`; unspecified dimensions will remain unchanged.

If the input dimensions are larger than the target crop dimensions, a centered cropping window will be extracted
from the input. The starting value for the cropping window is rounded down, which means that if the difference
between the input shape and the crop shape is odd, the cropping window will be shifted half a pixel to the left
of the input center.

If the input dimensions are smaller than the target crop dimensions, the input will be padded equally on both sides
to center it in the output. In cases where the total number of padding pixels is odd, an additional pixel will be
added to the right side.

The padding value used is zero.

## Attributes

- **axes** (INTS, optional): If provided, it specifies a subset of axes that 'shape' refer to. If not provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data). Negative value means counting dimensions from the back. Accepted range is [-r, r-1], where r = rank(data). Behavior is undefined if an axis is repeated.

## Inputs (2 - 2)

- **input_data** (T): Input to extract the centered crop from.
- **shape** (Tind): 1-D tensor representing the cropping window dimensions.

## Outputs (1 - 1)

- **output_data** (T): Output data.

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all tensor types.
- **Tind**: tensor(int32), tensor(int64)
  Constrain indices to integer types
