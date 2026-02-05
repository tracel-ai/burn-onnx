# Celu

First introduced in opset **12**

## Description

Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```

## Attributes

- **alpha** (FLOAT, optional): The Alpha value in Celu formula which control the shape of the unit. The default value is 1.0.

## Inputs (1 - 1)

- **X** (T): Input tensor

## Outputs (1 - 1)

- **Y** (T): Output tensor

## Type Constraints

- **T**: tensor(float)
  Constrain input and output types to float32 tensors.
