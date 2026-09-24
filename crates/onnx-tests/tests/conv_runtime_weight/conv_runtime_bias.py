#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.0",
#   "numpy",
# ]
# ///

# used to generate model: conv_runtime_bias.onnx
#
# A constant weight next to a bias that is a graph input, for Conv and for
# LayerNorm's scale. The bias must be read at run time, not dropped.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    f = TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info("x", f, [1, 2, 3, 3]),
        helper.make_tensor_value_info("conv_bias", f, [2]),
        helper.make_tensor_value_info("ln_bias", f, [3]),
    ]
    outputs = [
        helper.make_tensor_value_info("conv_out", f, [1, 2, 2, 2]),
        helper.make_tensor_value_info("ln_out", f, [1, 2, 3, 3]),
    ]
    weight = numpy_helper.from_array(
        (np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) * 0.1 - 0.7), "weight"
    )
    scale = numpy_helper.from_array(np.array([1.0, -0.5, 2.0], dtype=np.float32), "scale")
    nodes = [
        helper.make_node("Conv", ["x", "weight", "conv_bias"], ["conv_out"]),
        helper.make_node("LayerNormalization", ["x", "scale", "ln_bias"], ["ln_out"]),
    ]
    graph = helper.make_graph(
        nodes, "conv_runtime_bias", inputs, outputs, initializer=[weight, scale]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "conv_runtime_bias.onnx")

    feeds = {
        "x": np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3) * 0.2 - 1.0,
        "conv_bias": np.array([10.0, -20.0], dtype=np.float32),
        "ln_bias": np.array([5.0, 0.0, -5.0], dtype=np.float32),
    }
    conv_out, ln_out = ReferenceEvaluator(model).run(None, feeds)
    print(f"conv_out: {np.round(conv_out, 5).tolist()}")
    print(f"ln_out: {np.round(ln_out, 5).tolist()}")


if __name__ == "__main__":
    main()
