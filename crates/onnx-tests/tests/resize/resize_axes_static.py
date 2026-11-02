#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: resize_axes_static.onnx
#
# Resize with the `axes` attribute and constant sizes: only the width (axis 3) is
# resized, the height keeps its input size.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 6])
    sizes = numpy_helper.from_array(np.array([6], dtype=np.int64), name="sizes")

    node = helper.make_node(
        "Resize", ["x", "", "", "sizes"], ["y"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", axes=[3]
    )
    graph = helper.make_graph([node], "resize_axes_static_graph", [x], [y], [sizes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "resize_axes_static.onnx")

    ref = ReferenceEvaluator(model)
    data = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
    (out,) = ref.run(None, {"x": data})
    print(out.shape, out.tolist())


if __name__ == "__main__":
    main()
