# BlackmanWindow

First introduced in opset **17**

## Description

Generates a Blackman window as described in the paper https://ieeexplore.ieee.org/document/1455106.

## Attributes

- **output_datatype** (INT, optional): The data type of the output tensor. Strictly must be one of the values from DataType enum in TensorProto whose values correspond to T2. The default value is 1 = FLOAT.
- **periodic** (INT, optional): If 1, returns a window to be used as periodic function. If 0, return a symmetric window. When 'periodic' is specified, hann computes a window of length size + 1 and returns the first size points. The default value is 1.

## Inputs (1 - 1)

- **size** (T1): A scalar value indicating the length of the window.

## Outputs (1 - 1)

- **output** (T2): A Blackman window with length: size. The output has the shape: [size].

## Type Constraints

- **T1**: tensor(int32), tensor(int64)
  Constrain the input size to int64_t.
- **T2**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain output types to numeric tensors.
