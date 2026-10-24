#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: scatter_nd_bool_reductions.onnx
#
# ScatterND on bool data with every reduction. On bool, add and max are a logical or, mul
# and min a logical and (NumPy semantics). Each updated position differs from its update,
# so or, and and assign all disagree; indices are unique because burn documents duplicate
# targets as undefined for scatter_nd reductions, and index 3 is a true that stays untouched. Expected values from onnx.reference.ReferenceEvaluator, checked against a NumPy
# loop and, for add/mul, ONNX Runtime (which rejects max/min on bool).

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

REDUCTIONS = ["add", "mul", "max", "min"]


def main():
    inputs = [
        helper.make_tensor_value_info("data", TensorProto.BOOL, [4]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 1]),
        helper.make_tensor_value_info("updates", TensorProto.BOOL, [3]),
    ]
    nodes = [
        helper.make_node("ScatterND", ["data", "indices", "updates"], [f"y_{r}"], reduction=r)
        for r in REDUCTIONS
    ]
    outputs = [helper.make_tensor_value_info(f"y_{r}", TensorProto.BOOL, [4]) for r in REDUCTIONS]
    graph = helper.make_graph(nodes, "scatter_nd_bool_reductions_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_nd_bool_reductions.onnx")

    feed = {
        "data": np.array([False, True, True, True]),
        "indices": np.array([[0], [1], [2]], dtype=np.int64),
        "updates": np.array([True, False, True]),
    }
    results = ReferenceEvaluator(model).run(None, feed)
    for reduction, result in zip(REDUCTIONS, results):
        expected = feed["data"].copy()
        for (index,), value in zip(feed["indices"], feed["updates"]):
            if reduction in ("add", "max"):
                expected[index] = expected[index] or value
            else:
                expected[index] = expected[index] and value
        assert (result == expected).all(), reduction
        if reduction in ("add", "mul"):
            single = helper.make_model(
                helper.make_graph(
                    [helper.make_node("ScatterND", ["data", "indices", "updates"], ["y"], reduction=reduction)],
                    "g", inputs, [helper.make_tensor_value_info("y", TensorProto.BOOL, [4])],
                ),
                opset_imports=[helper.make_opsetid("", 18)],
            )
            single.ir_version = 8
            (runtime,) = ort.InferenceSession(single.SerializeToString()).run(None, feed)
            assert (runtime == result).all(), f"ONNX Runtime {reduction}"
        print(f"y_{reduction}", result.tolist())


if __name__ == "__main__":
    main()
