# SplitToSequence

First introduced in opset **11**

All versions: 11, 24

## Description

Split a tensor into a sequence of tensors, along the specified 'axis'.
Lengths of the parts can be specified using the optional argument 'split'.
If the argument `split' is not specified, a default scalar value of 1
is used as the value of `split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
if possible. The last chunk alone may be smaller than 'split' if the 'input' size
along the given axis 'axis' is not divisible by 'split'.
If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
in 'split' must be equal to the dimension size of input tensor on 'axis'.

## Attributes

- **axis** (INT, optional): Which axis to split on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1].
- **keepdims** (INT, optional): Keep the split dimension or not. Default 1, which means we keep split dimension. If input 'split' is specified, this attribute is ignored.

## Inputs (1 - 2)

- **input** (T): The tensor to split
- **split** (I, optional): Length of each output. It can be either a scalar(tensor of empty shape), or a 1-D tensor. All values must be >= 0.

## Outputs (1 - 1)

- **output_sequence** (S): One or more outputs forming a sequence of tensors after splitting

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input types to all tensor types.
- **I**: tensor(int32), tensor(int64)
  Constrain split size to integral tensor.
- **S**: seq(tensor(bfloat16)), seq(tensor(bool)), seq(tensor(complex128)), seq(tensor(complex64)), seq(tensor(double)), seq(tensor(float)), seq(tensor(float16)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(int8)), seq(tensor(string)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(uint8))
  Constrain output types to all tensor types.

## Version History

- **Opset 24**: Types: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 11**: Types: tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
