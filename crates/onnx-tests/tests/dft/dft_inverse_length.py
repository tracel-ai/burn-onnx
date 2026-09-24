#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/dft/dft_inverse_length.onnx
#
# Inverse complex DFT with a dft_length that truncates (8 -> 4) and one that
# zero-pads (8 -> 16). The result is scaled by 1/dft_length, not 1/N.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 2])
    truncated = helper.make_tensor_value_info("truncated", TensorProto.FLOAT, [1, 4, 2])
    padded = helper.make_tensor_value_info("padded", TensorProto.FLOAT, [1, 16, 2])

    def const(name, value):
        return helper.make_node(
            "Constant", [], [name],
            value=numpy_helper.from_array(np.array(value, dtype=np.int64), name=name),
        )

    nodes = [
        const("len4", 4),
        const("len16", 16),
        const("axis", 1),
        helper.make_node("DFT", ["x", "len4", "axis"], ["truncated"], inverse=1),
        helper.make_node("DFT", ["x", "len16", "axis"], ["padded"], inverse=1),
    ]
    graph = helper.make_graph(nodes, "dft_inverse_length", [x], [truncated, padded])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, "dft_inverse_length.onnx")

    xs = np.arange(16, dtype=np.float32).reshape(1, 8, 2) * 0.5 - 2.0
    for name, value in zip(
        ["truncated", "padded"], ReferenceEvaluator(model).run(None, {"x": xs})
    ):
        print(f"{name}: {np.round(value, 5).tolist()}")


if __name__ == "__main__":
    main()
