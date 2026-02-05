# CumSum

First introduced in opset **11**

All versions: 11, 14

## Description

Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 3, 6]
exclusive=1
output = [0, 1, 3]
exclusive=0
reverse=1
output = [6, 5, 3]
exclusive=1
reverse=1
output = [5, 3, 0]
```

## Attributes

- **exclusive** (INT, optional): If set to 1 will return exclusive sum in which the top element is not included. In other terms, if set to 1, the j-th output element would be the sum of the first (j-1) elements. Otherwise, it would be the sum of the first j elements.
- **reverse** (INT, optional): If set to 1 will perform the sums in reverse direction.

## Inputs (2 - 2)

- **x** (T): An input tensor that is to be processed.
- **axis** (T2): A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative value means counting dimensions from the back.

## Outputs (1 - 1)

- **y** (T): Output tensor of the same type as 'x' with cumulative sums of the x's elements

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
  Constrain input and output types to high-precision numeric tensors.
- **T2**: tensor(int32), tensor(int64)
  axis tensor can be int32 or int64 only

## Version History

- **Opset 14**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 11**: Types: tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
