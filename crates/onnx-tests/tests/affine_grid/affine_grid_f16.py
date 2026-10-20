#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: affine_grid_f16.onnx
#
# AffineGrid (opset 20) with a float16 theta and a runtime size. With size [1, 1, 1, 65536],
# a half-precision index would overflow (the largest f16 is 65504) and indices past 2048 are
# inexact, so the positions must be computed in a wider type before the cast to f16.
# Expected: x = (2 * i + 1) / W - 1 and y = 0 for an identity theta, rounded to f16.

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    theta = helper.make_tensor_value_info("theta", TensorProto.FLOAT16, [1, 2, 3])
    size = helper.make_tensor_value_info("size", TensorProto.INT64, [4])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT16, ["N", "H", "W", 2])
    node = helper.make_node("AffineGrid", ["theta", "size"], ["grid"], align_corners=0)
    graph = helper.make_graph([node], "affine_grid_f16_graph", [theta, size], [grid])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, "affine_grid_f16.onnx")

    width = 65536
    x = ((2 * np.arange(width) + 1) / width - 1).astype(np.float16)
    print("x[0], x[2047..2050], x[-1] in f16:", x[0], x[2047:2050].tolist(), x[-1])
    try:
        import onnxruntime as ort

        feed = {
            "theta": np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.float16),
            "size": np.array([1, 1, 1, width], dtype=np.int64),
        }
        (runtime,) = ort.InferenceSession(model.SerializeToString()).run(None, feed)
        print("ONNX Runtime max diff on x:", float(np.abs(runtime[0, 0, :, 0].astype(np.float64) - x).max()))
    except Exception as error:  # noqa: BLE001
        print("ONNX Runtime:", str(error).splitlines()[0][:150])


if __name__ == "__main__":
    main()
