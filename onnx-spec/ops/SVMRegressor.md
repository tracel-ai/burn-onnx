# SVMRegressor

Domain: **ai.onnx.ml**

First introduced in opset **1**

## Description

Support Vector Machine regression prediction and one-class SVM anomaly detection.

## Attributes

- **coefficients** (FLOATS, optional): Support vector coefficients.
- **kernel_params** (FLOATS, optional): List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.
- **kernel_type** (STRING, optional): The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.
- **n_supports** (INT, optional): The number of support vectors.
- **one_class** (INT, optional): Flag indicating whether the regression is a one-class SVM or not.
- **post_transform** (STRING, optional): Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'
- **rho** (FLOATS, optional)
- **support_vectors** (FLOATS, optional): Chosen support vectors

## Inputs (1 - 1)

- **X** (T): Data to be regressed.

## Outputs (1 - 1)

- **Y** (tensor(float)): Regression outputs (one score per target per example).

## Type Constraints

- **T**: tensor(double), tensor(float), tensor(int32), tensor(int64)
  The input type must be a tensor of a numeric type, either [C] or [N,C].
