# Imputer

Domain: **ai.onnx.ml** | Opset: **1**

Replaces values equal to `replaced_value_float`/`replaced_value_int64` with `imputed_value_floats`/`imputed_value_int64s`.
Length of imputed values must be 1 (broadcast) or F (one per feature, input shape `[*,F]`).

## Attributes

- **imputed_value_floats** (FLOATS, optional)
- **imputed_value_int64s** (INTS, optional)
- **replaced_value_float** (FLOAT, optional)
- **replaced_value_int64** (INT, optional)

## I/O

- **X** → **Y**: T (same shape and type)

## Type Constraints

- **T**: tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32), tensor(int64)
