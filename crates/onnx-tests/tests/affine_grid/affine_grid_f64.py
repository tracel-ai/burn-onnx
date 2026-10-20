#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: affine_grid_f64.onnx
#
# AffineGrid (opset 20) with a float64 theta and a runtime size: the grid must keep theta's
# dtype (T1). Expected values from ONNX Runtime, which returns float64; the ONNX reference
# implementation casts its result to float32, so it only agrees up to float32 precision.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    theta = helper.make_tensor_value_info("theta", TensorProto.DOUBLE, [2, 2, 3])
    size = helper.make_tensor_value_info("size", TensorProto.INT64, [4])
    grid = helper.make_tensor_value_info("grid", TensorProto.DOUBLE, ["N", "H", "W", 2])
    node = helper.make_node("AffineGrid", ["theta", "size"], ["grid"], align_corners=0)
    graph = helper.make_graph([node], "affine_grid_f64_graph", [theta, size], [grid])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, "affine_grid_f64.onnx")

    feed = {
        "theta": np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.5, -0.25, 0.1], [0.3, 0.8, -0.2]]],
            dtype=np.float64,
        ),
        "size": np.array([2, 1, 2, 3], dtype=np.int64),
    }
    (runtime,) = ort.InferenceSession(model.SerializeToString()).run(None, feed)
    (reference,) = ReferenceEvaluator(model).run(None, feed)
    print("grid", runtime.dtype, runtime.shape, runtime.tolist())
    print("max diff vs ONNX reference (float32):", float(np.abs(runtime - reference).max()))


if __name__ == "__main__":
    main()
