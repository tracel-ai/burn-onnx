#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/greater/greater_shape_rank_lift.onnx
#
# A Shape operand (dim 0 of x's shape, a length 1 Shape) compared with a rank 4 tensor.
# The Shape must be lifted to rank 4 before the comparison, in both operand orders.

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    x = helper.make_tensor_value_info("x", TensorProto.INT64, ["B", 3, 4, 5])
    out1 = helper.make_tensor_value_info("tensor_shape", TensorProto.BOOL, ["B", 3, 4, 5])
    out2 = helper.make_tensor_value_info("shape_tensor", TensorProto.BOOL, ["B", 3, 4, 5])
    idx = numpy_helper.from_array(np.array([0], dtype=np.int64), "idx")

    nodes = [
        helper.make_node("Shape", ["x"], ["shp"]),
        helper.make_node("Gather", ["shp", "idx"], ["dim0"], axis=0),
        helper.make_node("Greater", ["x", "dim0"], ["tensor_shape"]),
        helper.make_node("Greater", ["dim0", "x"], ["shape_tensor"]),
    ]
    graph = helper.make_graph(
        nodes, "greater_shape_rank_lift", [x], [out1, out2], initializer=[idx]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model)

    onnx_name = "greater_shape_rank_lift.onnx"
    onnx.save(model, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    test_x = np.arange(2 * 3 * 4 * 5, dtype=np.int64).reshape(2, 3, 4, 5)
    tensor_shape, shape_tensor = ReferenceEvaluator(model).run(None, {"x": test_x})
    print(f"tensor_shape: shape {tensor_shape.shape}\n{tensor_shape}")
    print(f"shape_tensor: shape {shape_tensor.shape}\n{shape_tensor}")


if __name__ == "__main__":
    main()
