#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: trilu_runtime_k.onnx
#
# Lower Trilu whose diagonal offset k is a graph input rather than a constant.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])
    k = helper.make_tensor_value_info("k", TensorProto.INT64, [])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])

    node = helper.make_node("Trilu", ["x", "k"], ["y"], upper=0)
    graph = helper.make_graph([node], "trilu_runtime_k_graph", [x, k], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "trilu_runtime_k.onnx")

    ref = ReferenceEvaluator(model)
    data = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    for kv in (-1, 2):
        [out] = ref.run(None, {"x": data, "k": np.array(kv, dtype=np.int64)})
        print(f"k={kv}: {out.tolist()}")


if __name__ == "__main__":
    main()
