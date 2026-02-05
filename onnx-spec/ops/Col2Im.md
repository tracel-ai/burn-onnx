# Col2Im

First introduced in opset **18**

## Description

The operator rearranges column blocks back into a multidimensional image

Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html,
but it only supports *batched* multi-dimensional image tensors.
Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.

NOTE:
  Although specifying image_shape looks redundant because it could be calculated from
  convolution formulas, it is required as input for more advanced scenarios as explained
  at PyTorch's implementation (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Col2Im.cpp#L10)

## Attributes

- **dilations** (INTS, optional): 1-dimensional tensor with dilation value along each spatial axis of the image. If not present, the dilation defaults to 1 along each spatial axis of the image.
- **pads** (INTS, optional): 1-dimensional tensor with padding value for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin is the number of pixels added at the beginning of axis `i` and xi_end is the number of pixels added at the end of axis `i`. If not present, the padding defaults to 0 along start and end of each spatial axis.
- **strides** (INTS, optional): 1-dimensional tensor with stride value along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

## Inputs (3 - 3)

- **input** (T): Input data tensor to be rearranged from column blocks back into an image. This is a 3-dimensional tensor containing [N, C * n-ary-product(block_shape), L], where N is batch dimension, C is image channel dimension and L is number of blocks.The blocks are enumerated in increasing lexicographic-order of their indices.For example, with an image-size 10*20 and block-size 9*18, there would be 2*3 blocks, enumerated in the order block(0, 0), block(0, 1), block(0, 2), block(1, 0), block(1, 1), block(1, 2).
- **image_shape** (tensor(int64)): The shape of the spatial dimensions of the image after rearranging the column blocks.This is a 1-dimensional tensor with size of at least 2, containing the value [H_img, W_img]  for a 2-D image or [dim_i1, dim_i2, ..., dim_iN] for a N-D image.
- **block_shape** (tensor(int64)): The shape of the block to apply on the input.This is a 1-dimensional tensor of size of at least 2, containing the value [H_block, W_block]  for a 2-D image or [dim_b1, dim_b2, ..., dim_bN] for a N-D block.This is the block-shape before dilation is applied to it.

## Outputs (1 - 1)

- **output** (T): Output tensor produced by rearranging blocks into an image.

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input and output types to all numeric tensor types.
