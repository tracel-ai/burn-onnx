#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: resize_keep_aspect_ratio_runtime.onnx
#
# Resize with runtime sizes (a graph input), no axes and keep_aspect_ratio_policy
# not_smaller. Without axes every axis is listed, so the scale also covers batch and
# channel: sizes [1, 1, 2, 3] on a 2x2 input -> scale max(1, 1, 1, 1.5) = 1.5 would
# resize the batch axis, while sizes [1, 1, 1, 2] -> scale 1 keeps the input size.
# Expected values from ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    sizes = helper.make_tensor_value_info("sizes", TensorProto.INT64, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, None, None])

    node = helper.make_node(
        "Resize", ["x", "", "", "sizes"], ["y"],
        mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", keep_aspect_ratio_policy="not_smaller",
    )
    graph = helper.make_graph([node], "resize_keep_aspect_ratio_runtime_graph", [x, sizes], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "resize_keep_aspect_ratio_runtime.onnx")

    session = ort.InferenceSession(model.SerializeToString())
    data = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    for s in ([1, 1, 1, 2], [1, 1, 2, 3]):
        (out,) = session.run(None, {"x": data, "sizes": np.array(s, dtype=np.int64)})
        print(s, out.shape, out.tolist())


if __name__ == "__main__":
    main()
