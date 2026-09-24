#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/dft/dft_complex.onnx
#
# Complex-input forward DFT, its inverse (which reconstructs the input), and the
# inverse DFT of a real input.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 2])
    r = helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 8, 1])
    outputs = [
        helper.make_tensor_value_info("spectrum", TensorProto.FLOAT, [1, 8, 2]),
        helper.make_tensor_value_info("roundtrip", TensorProto.FLOAT, [1, 8, 2]),
        helper.make_tensor_value_info("real_inverse", TensorProto.FLOAT, [1, 8, 2]),
    ]
    nodes = [
        helper.make_node("DFT", ["x"], ["spectrum"]),
        helper.make_node("DFT", ["spectrum"], ["roundtrip"], inverse=1),
        helper.make_node("DFT", ["r"], ["real_inverse"], inverse=1),
    ]
    graph = helper.make_graph(nodes, "dft_complex", [x, r], outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "dft_complex.onnx")

    xs = np.arange(16, dtype=np.float32).reshape(1, 8, 2) * 0.5 - 2.0
    rs = np.array([[[1], [2], [3], [4], [5], [6], [7], [8]]], dtype=np.float32)
    for name, value in zip(
        ["spectrum", "roundtrip", "real_inverse"],
        ReferenceEvaluator(model).run(None, {"x": xs, "r": rs}),
    ):
        print(f"{name}: {np.round(value, 5).tolist()}")


if __name__ == "__main__":
    main()
