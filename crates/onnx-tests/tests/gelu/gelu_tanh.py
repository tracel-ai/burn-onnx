#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/gelu/gelu_tanh.onnx
#
# Gelu with approximate="tanh". The inputs sit where the tanh approximation
# and the exact erf form differ by more than float rounding.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])

    node = helper.make_node("Gelu", inputs=["X"], outputs=["Y"], approximate="tanh")
    graph = helper.make_graph([node], "gelu_tanh_test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    onnx.checker.check_model(model)
    onnx.save(model, "gelu_tanh.onnx")

    test_input = np.array([-3.0, -1.5, 0.5, 2.0], dtype=np.float32)
    output = ReferenceEvaluator(model).run(None, {"X": test_input})[0]
    exact = ReferenceEvaluator(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("Gelu", ["X"], ["Y"])], "g", [X], [Y]
            ),
            opset_imports=[helper.make_opsetid("", 20)],
        )
    ).run(None, {"X": test_input})[0]
    print(f"tanh output: {output.tolist()}")
    print(f"exact output: {exact.tolist()}")


if __name__ == "__main__":
    main()
