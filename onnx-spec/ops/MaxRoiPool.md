# MaxRoiPool

First introduced in opset **1**

All versions: 1, 22

## Description

ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).

## Attributes

- **pooled_shape** (INTS, required): ROI pool output shape (height, width).
- **spatial_scale** (FLOAT, optional): Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.

## Inputs (2 - 2)

- **X** (T): Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
- **rois** (T): RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].

## Outputs (1 - 1)

- **Y** (T): RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).

## Type Constraints

- **T**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.

## Version History

- **Opset 22**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(float16)
