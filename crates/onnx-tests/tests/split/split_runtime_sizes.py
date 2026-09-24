#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: split_runtime_sizes.onnx
#
# Split into two outputs whose sizes arrive as a graph input.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6])
    split = helper.make_tensor_value_info("split", TensorProto.INT64, [2])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [None])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [None])

    node = helper.make_node("Split", ["x", "split"], ["a", "b"], axis=0)
    graph = helper.make_graph([node], "split_runtime_sizes_graph", [x, split], [a, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "split_runtime_sizes.onnx")

    ref = ReferenceEvaluator(model)
    data = np.arange(6, dtype=np.float32)
    for sizes in ([2, 4], [5, 1]):
        a_out, b_out = ref.run(None, {"x": data, "split": np.array(sizes, dtype=np.int64)})
        print(f"{sizes}: {a_out.tolist()} {b_out.tolist()}")


if __name__ == "__main__":
    main()
