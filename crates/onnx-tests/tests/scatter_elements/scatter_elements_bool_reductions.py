#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: scatter_elements_bool_reductions.onnx
#
# ScatterElements (axis 1) on bool data with every reduction. On bool, add and max are a
# logical or, mul and min a logical and (NumPy semantics). The indices repeat targets so
# the reduction matters. indices/updates match the data on the non-axis dimension (burn's
# native scatter); indices_small/updates_small have 1 row instead of 2 on the non-axis
# dimension, which selects the ScatterND fallback, with unique targets (burn documents
# duplicates as undefined for scatter_nd reductions) and row 1 untouched.
# Expected values from onnx.reference.ReferenceEvaluator (onnx 1.23.2; on 1.19 it assigned
# for mul), checked against a NumPy loop and, for the full-shape add/mul outputs, ONNX
# Runtime (which rejects max/min on bool).

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

REDUCTIONS = ["add", "mul", "max", "min"]


def reference_loop(data, indices, updates, reduction):
    output = data.copy()
    for row in range(indices.shape[0]):
        for col in range(indices.shape[1]):
            target = (row, indices[row, col])
            value = updates[row, col]
            if reduction in ("add", "max"):
                output[target] = output[target] or value
            else:
                output[target] = output[target] and value
    return output


def main():
    inputs = [
        helper.make_tensor_value_info("data", TensorProto.BOOL, [2, 3]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3]),
        helper.make_tensor_value_info("updates", TensorProto.BOOL, [2, 3]),
        helper.make_tensor_value_info("indices_small", TensorProto.INT64, [1, 3]),
        helper.make_tensor_value_info("updates_small", TensorProto.BOOL, [1, 3]),
    ]
    nodes, outputs = [], []
    for reduction in REDUCTIONS:
        name = f"y_{reduction}"
        nodes.append(helper.make_node("ScatterElements", ["data", "indices", "updates"], [name], axis=1, reduction=reduction))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.BOOL, [2, 3]))
    for reduction in ("add", "mul"):
        name = f"y_small_{reduction}"
        nodes.append(helper.make_node("ScatterElements", ["data", "indices_small", "updates_small"], [name], axis=1, reduction=reduction))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.BOOL, [2, 3]))
    graph = helper.make_graph(nodes, "scatter_elements_bool_reductions_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_elements_bool_reductions.onnx")

    feed = {
        "data": np.array([[False, True, False], [True, True, True]]),
        "indices": np.array([[0, 0, 2], [1, 1, 2]], dtype=np.int64),
        "updates": np.array([[True, True, False], [False, True, True]]),
        "indices_small": np.array([[0, 1, 2]], dtype=np.int64),
        "updates_small": np.array([[True, False, True]]),
    }
    results = dict(zip([o.name for o in outputs], ReferenceEvaluator(model).run(None, feed)))
    for reduction in REDUCTIONS:
        expected = reference_loop(feed["data"], feed["indices"], feed["updates"], reduction)
        assert (results[f"y_{reduction}"] == expected).all(), reduction
    for reduction in ("add", "mul"):
        expected = reference_loop(feed["data"], feed["indices_small"], feed["updates_small"], reduction)
        assert (results[f"y_small_{reduction}"] == expected).all(), reduction
        single = helper.make_model(
            helper.make_graph(
                [helper.make_node("ScatterElements", ["data", "indices", "updates"], ["y"], axis=1, reduction=reduction)],
                "g", inputs[:3], [helper.make_tensor_value_info("y", TensorProto.BOOL, [2, 3])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        single.ir_version = 8
        (runtime,) = ort.InferenceSession(single.SerializeToString()).run(
            None, {k: feed[k] for k in ("data", "indices", "updates")}
        )
        assert (runtime == results[f"y_{reduction}"]).all(), f"ONNX Runtime {reduction}"
    for name, value in results.items():
        print(name, value.tolist())


if __name__ == "__main__":
    main()
