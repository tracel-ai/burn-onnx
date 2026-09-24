#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: einsum_general.onnx
#
# Einsum forms beyond two-operand contractions: single-operand transpose,
# diagonal and trace, a batched diagonal through an ellipsis, a three-operand
# chain, and uppercase labels.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    inputs = [
        helper.make_tensor_value_info("sq", TensorProto.FLOAT, [3, 3]),
        helper.make_tensor_value_info("batch", TensorProto.FLOAT, [2, 3, 3]),
        helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4]),
        helper.make_tensor_value_info("c", TensorProto.FLOAT, [4, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info("transposed", TensorProto.FLOAT, [3, 2]),
        helper.make_tensor_value_info("diagonal", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("trace", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("batch_diagonal", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("chain", TensorProto.FLOAT, [2, 2]),
        helper.make_tensor_value_info("upper", TensorProto.FLOAT, [2, 4]),
    ]
    nodes = [
        helper.make_node("Einsum", ["a"], ["transposed"], equation="ij->ji"),
        helper.make_node("Einsum", ["sq"], ["diagonal"], equation="ii->i"),
        helper.make_node("Einsum", ["sq"], ["trace"], equation="ii"),
        helper.make_node("Einsum", ["batch"], ["batch_diagonal"], equation="...ii->...i"),
        helper.make_node("Einsum", ["a", "b", "c"], ["chain"], equation="ij,jk,kl->il"),
        helper.make_node("Einsum", ["a", "b"], ["upper"], equation="iJ,JK->iK"),
    ]

    graph = helper.make_graph(nodes, "einsum_general_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "einsum_general.onnx")

    feeds = {
        "sq": np.arange(9, dtype=np.float32).reshape(3, 3),
        "batch": np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.arange(12, dtype=np.float32).reshape(3, 4),
        "c": np.arange(8, dtype=np.float32).reshape(4, 2),
    }
    ref = ReferenceEvaluator(model)
    for name, value in zip([o.name for o in outputs], ref.run(None, feeds)):
        print(f"{name}: {value.tolist()}")


if __name__ == "__main__":
    main()
