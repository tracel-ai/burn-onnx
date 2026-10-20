#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: affine_grid_3d.onnx
#
# AffineGrid (opset 20) in 3D with theta [1, 3, 4] and a constant `size` [1, 1, 2, 3, 2].
# Two outputs: align_corners=0 and align_corners=1. Expected values from
# onnx.reference.ReferenceEvaluator, also checked against ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    theta = helper.make_tensor_value_info("theta", TensorProto.FLOAT, [1, 3, 4])
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2, 3, 2, 3])
        for name in ("grid", "grid_align_corners")
    ]
    size = numpy_helper.from_array(np.array([1, 1, 2, 3, 2], dtype=np.int64), name="size")
    nodes = [
        helper.make_node("AffineGrid", ["theta", "size"], ["grid"], align_corners=0),
        helper.make_node("AffineGrid", ["theta", "size"], ["grid_align_corners"], align_corners=1),
    ]
    graph = helper.make_graph(nodes, "affine_grid_3d_graph", [theta], outputs, [size])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, "affine_grid_3d.onnx")

    theta_data = np.array(
        [[[0.9, 0.1, 0.0, 0.2], [-0.1, 1.1, 0.3, 0.0], [0.0, 0.2, 0.7, -0.4]]], dtype=np.float32
    )
    feed = {"theta": theta_data}
    reference = ReferenceEvaluator(model).run(None, feed)
    runtime = ort.InferenceSession(model.SerializeToString()).run(None, feed)
    for name, ref, rt in zip(("grid", "grid_align_corners"), reference, runtime):
        print(name, ref.shape, np.round(ref, 6).tolist())
        print("  max diff vs ONNX Runtime:", float(np.abs(ref - rt).max()))


if __name__ == "__main__":
    main()
