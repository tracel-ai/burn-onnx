#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate OneHot model with axis=0 (inserts one-hot dim at front)."""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    depth_const = helper.make_node(
        "Constant", [], ["depth"],
        value=numpy_helper.from_array(np.array(4, dtype=np.int64), name="depth"),
    )
    values_const = helper.make_node(
        "Constant", [], ["values"],
        value=numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="values"),
    )
    one_hot_node = helper.make_node(
        "OneHot",
        ["indices", "depth", "values"],
        ["output"],
        axis=0,
    )

    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3])
    output = helper.make_tensor_value_info("output", TensorProto.INT64, [4, 3])

    graph = helper.make_graph(
        [depth_const, values_const, one_hot_node],
        "one_hot_axis0", [indices], [output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])
    onnx.checker.check_model(model)
    onnx.save(model, "one_hot_axis0.onnx")

    # Compute expected outputs
    indices_val = np.array([0, 2, 3], dtype=np.int64)
    ref = ReferenceEvaluator(model)
    (output_val,) = ref.run(None, {"indices": indices_val})
    print("Input indices:", repr(indices_val))
    print("\nExpected output:")
    print(repr(output_val))


if __name__ == "__main__":
    main()
