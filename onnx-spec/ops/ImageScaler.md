# ImageScaler

Legacy experimental operator. All versions: 1

Source: https://github.com/onnx/onnx/blob/v1.3.0/docs/Operators.md#ImageScaler

Scale and bias an input image in NCHW format. The output has the input shape and type.

Attributes:
- `scale`: float multiplier, default 1.0.
- `bias`: list of floats, one bias per channel, in channel order. Omitted bias adds nothing.

Input/output types: float16, float32, float64.

This experimental operator was removed from modern ONNX schemas. Legacy models such as
TinyYOLOv2 still contain it. Tests use an equivalent Mul/Add graph as a reference.
