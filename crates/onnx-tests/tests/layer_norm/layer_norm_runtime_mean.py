#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: layer_norm_runtime_mean.onnx
#
# LayerNormalization over the last two axes with the scale as a graph input, no
# bias, and only the optional Mean output requested.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 3])
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 3])
    mean = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [2, 1, 1])

    node = helper.make_node(
        "LayerNormalization", ["x", "scale"], ["y", "mean"], axis=1
    )
    graph = helper.make_graph([node], "layer_norm_runtime_mean", [x, scale], [y, mean])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "layer_norm_runtime_mean.onnx")

    data = np.arange(12, dtype=np.float32).reshape(2, 2, 3) ** 1.5
    scale_data = np.array([[1.0, 2.0, 0.5], [-1.0, 1.5, 3.0]], dtype=np.float32)
    y_out, mean_out = ReferenceEvaluator(model).run(None, {"x": data, "scale": scale_data})
    print(f"y: {y_out.tolist()}")
    print(f"mean: {mean_out.tolist()}")


if __name__ == "__main__":
    main()
