#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.0",
#   "numpy",
# ]
# ///

# used to generate model: layer_norm_broadcast.onnx
#
# LayerNormalization over the last two axes (axis=1 on a rank-3 input) with a
# constant 1-D scale and a [1, D] bias, both broadcasting against the normalized
# axes. A 1-D constant scale must not be taken as normalizing the last axis only.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 3])
    scale = numpy_helper.from_array(np.array([1.0, 2.0, -1.0], dtype=np.float32), "scale")
    bias = numpy_helper.from_array(np.array([[0.5, 0.0, -0.5]], dtype=np.float32), "bias")
    node = helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["y"], axis=1)
    graph = helper.make_graph(
        [node], "layer_norm_broadcast", [x], [y], initializer=[scale, bias]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "layer_norm_broadcast.onnx")

    data = np.arange(12, dtype=np.float32).reshape(2, 2, 3) ** 1.5
    [out] = ReferenceEvaluator(model).run(None, {"x": data})
    print(f"y: {np.round(out, 6).tolist()}")


if __name__ == "__main__":
    main()
