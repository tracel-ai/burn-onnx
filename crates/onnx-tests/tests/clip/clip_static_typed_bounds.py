#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: clip_static_typed_bounds.onnx
#
# Clip with constant bounds that were mishandled when read through f64:
# - uint32 data with uint32 min/max (bounds of this dtype used to be dropped)
# - int64 data with max = 2^53 + 1, which f64 rounds down to 2^53

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

BIG = 2**53 + 1


def const(name, value, dtype):
    return helper.make_node(
        "Constant", [], [name], value=numpy_helper.from_array(np.array(value, dtype=dtype))
    )


nodes = [
    const("min_u", 2, np.uint32),
    const("max_u", 7, np.uint32),
    helper.make_node("Clip", ["x_u", "min_u", "max_u"], ["y_u"]),
    const("max_i", BIG, np.int64),
    helper.make_node("Clip", ["x_i", "", "max_i"], ["y_i"]),
]
graph = helper.make_graph(
    nodes,
    "main_graph",
    [
        helper.make_tensor_value_info("x_u", TensorProto.UINT32, [3]),
        helper.make_tensor_value_info("x_i", TensorProto.INT64, [3]),
    ],
    [
        helper.make_tensor_value_info("y_u", TensorProto.UINT32, [3]),
        helper.make_tensor_value_info("y_i", TensorProto.INT64, [3]),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
onnx.checker.check_model(model)
onnx.save(model, "clip_static_typed_bounds.onnx")

x_u = np.array([0, 5, 10], dtype=np.uint32)
x_i = np.array([0, BIG, BIG + 2], dtype=np.int64)
y_u, y_i = ReferenceEvaluator(model).run(None, {"x_u": x_u, "x_i": x_i})
print(f"y_u: {y_u.tolist()}")
print(f"y_i: {y_i.tolist()}")
