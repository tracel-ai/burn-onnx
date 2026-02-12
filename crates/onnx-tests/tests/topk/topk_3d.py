#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate TopK model with axis=1 on a [2, 4, 3] tensor, k=2."""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    k_const = helper.make_node(
        "Constant", [], ["K"],
        value=numpy_helper.from_array(np.array([2], dtype=np.int64), name="K"),
    )
    topk_node = helper.make_node(
        "TopK", ["X", "K"], ["Values", "Indices"],
        axis=1, largest=1, sorted=1,
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 3])
    Values = helper.make_tensor_value_info("Values", TensorProto.FLOAT, [2, 2, 3])
    Indices = helper.make_tensor_value_info("Indices", TensorProto.INT64, [2, 2, 3])

    graph = helper.make_graph([k_const, topk_node], "topk_3d", [X], [Values, Indices])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])
    onnx.checker.check_model(model)
    onnx.save(model, "topk_3d.onnx")

    # Compute expected outputs
    np.random.seed(42)
    x = np.random.randn(2, 4, 3).astype(np.float32)
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
