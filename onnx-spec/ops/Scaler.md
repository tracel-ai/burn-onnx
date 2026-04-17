# Scaler

Domain: **ai.onnx.ml**

First introduced in opset **1**

## Description

Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

## Attributes

- **offset** (FLOATS, optional): First, offset by this.<br>Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.
- **scale** (FLOATS, optional): Second, multiply by this.<br>Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.<br>Must be same length as 'offset'

## Inputs (1 - 1)

- **X** (T): Data to be scaled.

## Outputs (1 - 1)

- **Y** (tensor(float)): Scaled output data.

## Type Constraints

- **T**: tensor(double), tensor(float), tensor(int32), tensor(int64)
  The input must be a tensor of a numeric type.
