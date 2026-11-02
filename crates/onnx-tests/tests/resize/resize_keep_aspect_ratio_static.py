#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: resize_keep_aspect_ratio_static.onnx
#
# Resize with constant sizes [7, 8] on axes [2, 3] and keep_aspect_ratio_policy, so
# the adjusted size is computed at import time: not_larger -> scale 3.5 -> 7x7,
# not_smaller -> scale 4 -> 8x8. Expected values from ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y_not_larger = helper.make_tensor_value_info("y_not_larger", TensorProto.FLOAT, [1, 1, 7, 7])
    y_not_smaller = helper.make_tensor_value_info("y_not_smaller", TensorProto.FLOAT, [1, 1, 8, 8])
    sizes = numpy_helper.from_array(np.array([7, 8], dtype=np.int64), name="sizes")

    nodes = [
        helper.make_node(
            "Resize", ["x", "", "", "sizes"], [output],
            mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", axes=[2, 3], keep_aspect_ratio_policy=policy,
        )
        for output, policy in [("y_not_larger", "not_larger"), ("y_not_smaller", "not_smaller")]
    ]
    graph = helper.make_graph(
        nodes, "resize_keep_aspect_ratio_static_graph", [x], [y_not_larger, y_not_smaller], [sizes]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "resize_keep_aspect_ratio_static.onnx")

    data = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    not_larger, not_smaller = ort.InferenceSession(model.SerializeToString()).run(None, {"x": data})
    print("not_larger", not_larger.shape, not_larger.tolist())
    print("not_smaller", not_smaller.shape, not_smaller.tolist())


if __name__ == "__main__":
    main()
