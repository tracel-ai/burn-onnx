#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate Split model with explicit split sizes on axis=1."""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    k_const = helper.make_node(
        "Constant", [], ["split_sizes"],
        value=numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="split_sizes"),
    )
    split_node = helper.make_node(
        "Split",
        ["X", "split_sizes"],
        ["Y0", "Y1"],
        axis=1,
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])
    Y0 = helper.make_tensor_value_info("Y0", TensorProto.FLOAT, [3, 2])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 3])

    graph = helper.make_graph([k_const, split_node], "split_axis1", [X], [Y0, Y1])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])
    onnx.checker.check_model(model)
    onnx.save(model, "split_axis1.onnx")

    # Compute expected outputs
    np.random.seed(42)
    x = np.random.randn(3, 5).astype(np.float32)
    ref = ReferenceEvaluator(model)
    y0, y1 = ref.run(None, {"X": x})
    print("Input X:")
    print(repr(x))
    print(f"\nY0 shape: {y0.shape}")
    print(repr(y0))
    print(f"\nY1 shape: {y1.shape}")
    print(repr(y1))


if __name__ == "__main__":
    main()
