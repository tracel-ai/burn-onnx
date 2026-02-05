# ReduceLogSumExp

First introduced in opset **1**

All versions: 1, 11, 13, 18

## Description

Computes the log sum exponent of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

## Attributes

- **keepdims** (INT, optional): Keep the reduced dimension or not, default 1 means keep reduced dimension.
- **noop_with_empty_axes** (INT, optional): Defines behavior when axes is not provided or is empty. If false (default), reduction happens over all axes. If true, no reduction is applied, but other operations will be performed. For example, ReduceSumSquare acts as a vanilla Square.

## Inputs (1 - 2)

- **data** (T): An input tensor.
- **axes** (tensor(int64), optional): Optional input list of integers, along which to reduce. The default is to reduce over empty axes. When axes is empty (either not provided or explicitly empty), behavior depends on 'noop_with_empty_axes': reduction over all axes if 'noop_with_empty_axes' is false, or no reduction is applied if 'noop_with_empty_axes' is true (but other operations will be performed). Accepted range is [-r, r-1] where r = rank(data).

## Outputs (1 - 1)

- **reduced** (T): Reduced output tensor.

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
  Constrain input and output types to numeric tensors.

## Version History

- **Opset 18**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 13**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 11**: Types: tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
