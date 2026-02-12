#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate TopK model where k equals the full dimension size (k=5 on axis=1, shape [3, 5])."""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    k_const = helper.make_node(
        "Constant", [], ["K"],
        value=numpy_helper.from_array(np.array([5], dtype=np.int64), name="K"),
    )
    topk_node = helper.make_node(
        "TopK", ["X", "K"], ["Values", "Indices"],
        axis=1, largest=1, sorted=1,
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])
    Values = helper.make_tensor_value_info("Values", TensorProto.FLOAT, [3, 5])
    Indices = helper.make_tensor_value_info("Indices", TensorProto.INT64, [3, 5])

    graph = helper.make_graph([k_const, topk_node], "topk_k_full", [X], [Values, Indices])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])
    onnx.checker.check_model(model)
    onnx.save(model, "topk_k_full.onnx")

    # Compute expected outputs
    np.random.seed(42)
    x = np.random.randn(3, 5).astype(np.float32)
    ref = ReferenceEvaluator(model)
    values, indices = ref.run(None, {"X": x})
    print("Input X:")
    print(repr(x))
    print("\nExpected Values:")
    print(repr(values))
    print("\nExpected Indices:")
    print(repr(indices))


if __name__ == "__main__":
    main()
