#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: global_max_pool.onnx
#
# GlobalMaxPool over a rank-3 and a rank-4 input.

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    x3 = helper.make_tensor_value_info("x3", TensorProto.FLOAT, [1, 2, 4])
    x4 = helper.make_tensor_value_info("x4", TensorProto.FLOAT, [2, 2, 2, 3])
    y3 = helper.make_tensor_value_info("y3", TensorProto.FLOAT, [1, 2, 1])
    y4 = helper.make_tensor_value_info("y4", TensorProto.FLOAT, [2, 2, 1, 1])

    nodes = [
        helper.make_node("GlobalMaxPool", ["x3"], ["y3"]),
        helper.make_node("GlobalMaxPool", ["x4"], ["y4"]),
    ]
    graph = helper.make_graph(nodes, "global_max_pool_graph", [x3, x4], [y3, y4])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "global_max_pool.onnx")

    np.random.seed(42)
    feeds = {
        "x3": np.random.randn(1, 2, 4).astype(np.float32),
        "x4": np.random.randn(2, 2, 2, 3).astype(np.float32),
    }
    # onnx.reference reduces the wrong axes for rank-3 GlobalMaxPool, so the
    # expected outputs come from numpy: the max over every spatial axis.
    y3_out = feeds["x3"].max(axis=2, keepdims=True)
    y4_out = feeds["x4"].max(axis=(2, 3), keepdims=True)
    for name, value in feeds.items():
        print(f"{name}: {value.tolist()}")
    print(f"y3: {y3_out.tolist()}")
    print(f"y4: {y4_out.tolist()}")


if __name__ == "__main__":
    main()
