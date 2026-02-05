# NonMaxSuppression

First introduced in opset **10**

All versions: 10, 11

## Description

Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.

## Attributes

- **center_point_box** (INT, optional): Integer indicate the format of the box data. The default is 0. 0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners and the coordinates can be provided as normalized (i.e., lying in the interval [0, 1]) or absolute. Mostly used for TF models. 1 - the box data is supplied as [x_center, y_center, width, height]. Mostly used for Pytorch models.

## Inputs (2 - 5)

- **boxes** (tensor(float)): An input tensor with shape [num_batches, spatial_dimension, 4]. The single box data format is indicated by center_point_box.
- **scores** (tensor(float)): An input tensor with shape [num_batches, num_classes, spatial_dimension]
- **max_output_boxes_per_class** (tensor(int64), optional): Integer representing the maximum number of boxes to be selected per batch per class. It is a scalar. Default to 0, which means no output.
- **iou_threshold** (tensor(float), optional): Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1]. Default to 0.
- **score_threshold** (tensor(float), optional): Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.

## Outputs (1 - 1)

- **selected_indices** (tensor(int64)): selected indices from the boxes tensor. [num_selected_indices, 3], the selected index format is [batch_index, class_index, box_index].

## Version History

- **Opset 11**:
- **Opset 10**:
