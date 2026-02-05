# DynamicQuantizeLinear

First introduced in opset **11**

## Description

A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
```

* where qmax and qmin are max and min values for quantization range i.e. [0, 255] in case of uint8
* data range is adjusted to include 0.

Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
```

* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.

Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
```

* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.

## Inputs (1 - 1)

- **x** (T1): Input tensor

## Outputs (3 - 3)

- **y** (T2): Quantized output tensor
- **y_scale** (tensor(float)): Output scale. It's a scalar, which means a per-tensor/layer quantization.
- **y_zero_point** (T2): Output zero point. It's a scalar, which means a per-tensor/layer quantization.

## Type Constraints

- **T1**: tensor(float)
  Constrain 'x' to float tensor.
- **T2**: tensor(uint8)
  Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.
