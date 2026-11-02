#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: resize_axes_runtime.onnx
#
# Resize with the `axes` attribute in reverse order (width, then height) and scales
# that arrive as a graph input, so they are only known at run time.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, None, None])

    node = helper.make_node(
        "Resize", ["x", "", "scales"], ["y"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", axes=[3, 2]
    )
    graph = helper.make_graph([node], "resize_axes_runtime_graph", [x, scales], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "resize_axes_runtime.onnx")

    ref = ReferenceEvaluator(model)
    data = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    # Width x3, height x2
    (out,) = ref.run(None, {"x": data, "scales": np.array([3.0, 2.0], dtype=np.float32)})
    print(out.shape, out.tolist())


if __name__ == "__main__":
    main()
