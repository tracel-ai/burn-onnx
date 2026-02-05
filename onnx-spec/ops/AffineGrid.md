# AffineGrid

First introduced in opset **20**

## Description

Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
(https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
An affine matrix `theta` is applied to a position tensor represented in its homogeneous expression. Here is an example in 3D:
```
[r00, r01, r02, t0]   [x]   [x']
[r10, r11, r12, t1] * [y] = [y']
[r20, r21, r22, t2]   [z]   [z']
[0,   0,   0,   1 ]   [1]   [1 ]
```
where `(x, y, z)` is the position in the original space, `(x', y', z')` is the position in the output space.
The last row is always `[0, 0, 0, 1]` and is not stored in the affine matrix. Therefore we have `theta` of shape `(N, 2, 3)` for 2D or `(N, 3, 4)` for 3D.

Input `size` is used to define grid of positions evenly spaced in the original 2D or 3D space, with dimensions ranging from `-1` to `1`.
The output `grid` contains positions in the output space.

When `align_corners=1`, consider `-1` and `1` to refer to the centers of the corner pixels (mark `v` in illustration).
```
v            v            v            v
|-------------------|------------------|
-1                  0                  1
```
When `align_corners=0`, consider `-1` and `1` to refer to the outer edge of the corner pixels.
```
    v        v         v         v
|------------------|-------------------|
-1                 0                   1
```

## Attributes

- **align_corners** (INT, optional): if align_corners=1, consider -1 and 1 to refer to the centers of the corner pixels. if align_corners=0, consider -1 and 1 to refer to the outer edge the corner pixels.

## Inputs (2 - 2)

- **theta** (T1): input batch of affine matrices with shape (N, 2, 3) for 2D or (N, 3, 4) for 3D
- **size** (T2): the target output image size (N, C, H, W) for 2D or (N, C, D, H, W) for 3D

## Outputs (1 - 1)

- **grid** (T1): output tensor of shape (N, H, W, 2) of 2D sample coordinates or (N, D, H, W, 3) of 3D sample coordinates.

## Type Constraints

- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain grid types to float tensors.
- **T2**: tensor(int64)
  Constrain size's type to int64 tensors.
