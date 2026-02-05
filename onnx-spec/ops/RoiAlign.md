# RoiAlign

First introduced in opset **10**

All versions: 10, 16, 22

## Description

Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.

## Attributes

- **coordinate_transformation_mode** (STRING, optional): Allowed values are 'half_pixel' and 'output_half_pixel'. Use the value 'half_pixel' to pixel shift the input coordinates by -0.5 (the recommended behavior). Use the value 'output_half_pixel' to omit the pixel shift for the input (use this for a backward-compatible behavior).
- **mode** (STRING, optional): The pooling method. Two modes are supported: 'avg' and 'max'. Default is 'avg'.
- **output_height** (INT, optional): default 1; Pooled output Y's height.
- **output_width** (INT, optional): default 1; Pooled output Y's width.
- **sampling_ratio** (INT, optional): Number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. If == 0, then an adaptive number of grid points are used (computed as ceil(roi_width / output_width), and likewise for height). Default is 0.
- **spatial_scale** (FLOAT, optional): Multiplicative spatial scale factor to translate ROI coordinates from their input spatial scale to the scale used when pooling, i.e., spatial scale of the input feature map X relative to the input image. E.g.; default is 1.0f.

## Inputs (3 - 3)

- **X** (T1): Input data tensor from the previous operator; 4-D feature map of shape (N, C, H, W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
- **rois** (T1): RoIs (Regions of Interest) to pool over; rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...]. The RoIs' coordinates are in the coordinate system of the input image. Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.
- **batch_indices** (T2): 1-D tensor of shape (num_rois,) with each element denoting the index of the corresponding image in the batch.

## Outputs (1 - 1)

- **Y** (T1): RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI X[r-1].

## Type Constraints

- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain types to float tensors.
- **T2**: tensor(int64)
  Constrain types to int tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 16**: Types: tensor(double), tensor(float), tensor(float16)
- **Opset 10**: Types: tensor(double), tensor(float), tensor(float16)
