#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: global_avr_pool_squeeze.onnx
#
# GlobalAveragePool followed by a Squeeze with no axes, the usual classifier head.
# Squeeze without axes drops every dim of size 1, so burn-onnx can only infer its
# output rank if GlobalAveragePool's static shape reports the spatial dims as 1.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Squeeze", ["pooled"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "global_avr_pool_squeeze",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 5])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "global_avr_pool_squeeze.onnx")

    x = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    (out,) = ReferenceEvaluator(model).run(None, {"input": x})
    print(f"output shape: {out.shape}")
    print(f"output: {out.tolist()}")


if __name__ == "__main__":
    main()
