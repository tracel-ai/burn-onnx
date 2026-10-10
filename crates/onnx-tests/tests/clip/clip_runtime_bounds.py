#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: clip_runtime_bounds.onnx
#
# Clip with min/max that arrive as scalar graph inputs (runtime bounds), on a float
# tensor (bounds cast to f64) and on an int64 tensor (bounds cast to i64), plus a
# min-only and a max-only clip.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("min_f", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("max_f", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("x_i", TensorProto.INT64, [4]),
        helper.make_tensor_value_info("min_i", TensorProto.INT64, []),
        helper.make_tensor_value_info("max_i", TensorProto.INT64, []),
    ]
    outputs = [
        helper.make_tensor_value_info("both_f", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("min_only_f", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("max_only_i", TensorProto.INT64, [4]),
        helper.make_tensor_value_info("both_i", TensorProto.INT64, [4]),
    ]
    nodes = [
        helper.make_node("Clip", ["x", "min_f", "max_f"], ["both_f"]),
        helper.make_node("Clip", ["x", "min_f", ""], ["min_only_f"]),
        helper.make_node("Clip", ["x_i", "", "max_i"], ["max_only_i"]),
        helper.make_node("Clip", ["x_i", "min_i", "max_i"], ["both_i"]),
    ]
    graph = helper.make_graph(nodes, "clip_runtime_bounds_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "clip_runtime_bounds.onnx")

    ref = ReferenceEvaluator(model)
    feeds = {
        "x": np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32),
        "min_f": np.array(-1.0, dtype=np.float32),
        "max_f": np.array(1.0, dtype=np.float32),
        "x_i": np.array([-5, -1, 1, 5], dtype=np.int64),
        "min_i": np.array(-2, dtype=np.int64),
        "max_i": np.array(3, dtype=np.int64),
    }
    for name, value in zip(["both_f", "min_only_f", "max_only_i", "both_i"], ref.run(None, feeds)):
        print(name, value.tolist())


if __name__ == "__main__":
    main()
