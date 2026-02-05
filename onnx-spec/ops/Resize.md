# Resize

First introduced in opset **10**

All versions: 10, 11, 13, 18, 19

## Description

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
```
output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
```
if input \"sizes\" is not specified.

## Attributes

- **antialias** (INT, optional): If set to 1, "linear" and "cubic" interpolation modes will use an antialiasing filter when downscaling. Antialiasing is achieved by stretching the resampling filter by a factor max(1, 1 / scale), which means that when downsampling, more input pixels contribute to an output pixel.
- **axes** (INTS, optional): If provided, it specifies a subset of axes that 'roi', 'scales' and 'sizes' refer to. If not provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data). Non-specified dimensions are interpreted as non-resizable. Negative value means counting dimensions from the back. Accepted range is [-r, r-1], where r = rank(data). Behavior is undefined if an axis is repeated.
- **coordinate_transformation_mode** (STRING, optional): This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.
- **cubic_coeff_a** (FLOAT, optional): The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. This attribute is valid only if mode is "cubic".
- **exclude_outside** (INT, optional): If set to 1, the weight of sampling locations outside the tensor will be set to 0 and the weight will be renormalized so that their sum is 1.0. The default value is 0.
- **extrapolation_value** (FLOAT, optional): When coordinate_transformation_mode is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f.
- **keep_aspect_ratio_policy** (STRING, optional): This attribute describes how to interpret the `sizes` input with regard to keeping the original aspect ratio of the input, and it is not applicable when
- **mode** (STRING, optional): Three interpolation modes: "nearest" (default), "linear" and "cubic". The "linear" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). The "cubic" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).
- **nearest_mode** (STRING, optional): Four modes: "round_prefer_floor" (default, as known as round half down), "round_prefer_ceil" (as known as round half up), "floor", "ceil". Only used by nearest interpolation. It indicates how to get "nearest" pixel in input tensor from x_original, so this attribute is valid only if "mode" is "nearest".

## Inputs (1 - 4)

- **X** (T1): N-D tensor
- **roi** (T2, optional): 1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X or the length of axes, if provided. The RoIs' coordinates are normalized in the coordinate system of the input image. It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
- **scales** (tensor(float), optional): The scale array along each dimension. It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X' or the length of 'axes', if provided. One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified. If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.
- **sizes** (tensor(int64), optional): Target size of the output tensor. Its interpretation depends on the 'keep_aspect_ratio_policy' value.The number of elements of 'sizes' should be the same as the rank of input 'X', or the length of 'axes', if provided. Only one of 'scales' and 'sizes' can be specified.

## Outputs (1 - 1)

- **Y** (T1): N-D tensor after resizing

## Type Constraints

- **T1**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input 'X' and output 'Y' to all tensor types.
- **T2**: tensor(double), tensor(float), tensor(float16)
  Constrain roi type to float or double.

## Version History

- **Opset 19**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 18**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 13**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 11**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 10**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
