#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: resize_keep_aspect_ratio_halfway.onnx
#
# keep_aspect_ratio_policy not_larger on a 25x50 input with sizes [15, 29] on axes [2, 3]:
# scale = min(15/25, 29/50) = 0.58 and 0.58 * 25 = 14.5, which rounds half up to 15. In f64,
# 29.0 / 50.0 * 25.0 is 14.499999999999998, so float rounding would give 14. The same sizes
# are applied as a constant (adjusted at import) and as a graph input (adjusted at run
# time). Expected shapes from ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 25, 50])
    sizes = helper.make_tensor_value_info("sizes", TensorProto.INT64, [2])
    y_static = helper.make_tensor_value_info("y_static", TensorProto.FLOAT, [1, 1, 15, 29])
    y_runtime = helper.make_tensor_value_info("y_runtime", TensorProto.FLOAT, [1, 1, 15, 29])
    constant_sizes = numpy_helper.from_array(np.array([15, 29], dtype=np.int64), name="constant_sizes")

    attrs = dict(mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", axes=[2, 3], keep_aspect_ratio_policy="not_larger")
    nodes = [
        helper.make_node("Resize", ["x", "", "", "constant_sizes"], ["y_static"], **attrs),
        helper.make_node("Resize", ["x", "", "", "sizes"], ["y_runtime"], **attrs),
    ]
    graph = helper.make_graph(
        nodes, "resize_keep_aspect_ratio_halfway_graph", [x, sizes], [y_static, y_runtime],
        [constant_sizes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "resize_keep_aspect_ratio_halfway.onnx")

    data = np.zeros((1, 1, 25, 50), dtype=np.float32)
    static, runtime = ort.InferenceSession(model.SerializeToString()).run(
        None, {"x": data, "sizes": np.array([15, 29], dtype=np.int64)}
    )
    print("y_static", static.shape)
    print("y_runtime", runtime.shape)


if __name__ == "__main__":
    main()
