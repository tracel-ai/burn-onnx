#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: affine_grid_2d.onnx
#
# AffineGrid (opset 20) with theta [2, 2, 3] and `size` as a graph input, as in the official
# tests, so the grid is built at run time. Two outputs: align_corners=0 and align_corners=1.
# Expected values from onnx.reference.ReferenceEvaluator, also checked against ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    theta = helper.make_tensor_value_info("theta", TensorProto.FLOAT, [2, 2, 3])
    size = helper.make_tensor_value_info("size", TensorProto.INT64, [4])
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N", "H", "W", 2])
        for name in ("grid", "grid_align_corners")
    ]
    nodes = [
        helper.make_node("AffineGrid", ["theta", "size"], ["grid"], align_corners=0),
        helper.make_node("AffineGrid", ["theta", "size"], ["grid_align_corners"], align_corners=1),
    ]
    graph = helper.make_graph(nodes, "affine_grid_2d_graph", [theta, size], outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, "affine_grid_2d.onnx")

    theta_data = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.5, -0.25, 0.1], [0.3, 0.8, -0.2]]],
        dtype=np.float32,
    )
    for size_data in ([2, 1, 3, 4], [2, 1, 1, 3]):
        feed = {"theta": theta_data, "size": np.array(size_data, dtype=np.int64)}
        reference = ReferenceEvaluator(model).run(None, feed)
        runtime = ort.InferenceSession(model.SerializeToString()).run(None, feed)
        print("size", size_data)
        for name, ref, rt in zip(("grid", "grid_align_corners"), reference, runtime):
            print(" ", name, ref.shape, np.round(ref, 6).tolist())
            print("    max diff vs ONNX Runtime:", float(np.abs(ref - rt).max()))


if __name__ == "__main__":
    main()
