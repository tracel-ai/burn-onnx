#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/dft/dft_length.onnx
#
# Real DFT with an explicit dft_length: a length-5 signal zero-padded to 8
# (onesided) and truncated to 4 (full spectrum).

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 5, 1])
    padded = helper.make_tensor_value_info("padded", TensorProto.FLOAT, [1, 5, 2])
    truncated = helper.make_tensor_value_info("truncated", TensorProto.FLOAT, [1, 4, 2])

    def const(name, value):
        return helper.make_node(
            "Constant", [], [name],
            value=numpy_helper.from_array(np.array(value, dtype=np.int64), name=name),
        )

    nodes = [
        const("len8", 8),
        const("len4", 4),
        const("axis", 1),
        helper.make_node("DFT", ["input", "len8", "axis"], ["padded"], onesided=1),
        helper.make_node("DFT", ["input", "len4", "axis"], ["truncated"], onesided=0),
    ]
    graph = helper.make_graph(nodes, "dft_length_model", [X], [padded, truncated])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, "dft_length.onnx")

    test_input = np.array([[[1], [2], [3], [4], [5]]], dtype=np.float32)
    padded_out, truncated_out = ReferenceEvaluator(model).run(None, {"input": test_input})
    print(f"padded: {padded_out.tolist()}")
    print(f"truncated: {truncated_out.tolist()}")


if __name__ == "__main__":
    main()
