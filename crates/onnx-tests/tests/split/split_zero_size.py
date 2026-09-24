#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: split_zero_size.onnx
#
# Split with zero-length parts: constant sizes [3, 0, 2], runtime sizes, and
# num_outputs=4 over an axis of length 3 (chunks 1, 1, 1, 0). Every output must
# exist, including the empty ones.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 18


def main():
    sizes = helper.make_node(
        "Constant",
        [],
        ["sizes"],
        value=numpy_helper.from_array(np.array([3, 0, 2], dtype=np.int64)),
    )
    static_split = helper.make_node(
        "Split", ["x", "sizes"], ["s0", "s1", "s2"], axis=0
    )
    runtime_split = helper.make_node(
        "Split", ["x", "runtime_sizes"], ["r0", "r1", "r2"], axis=0
    )
    count_split = helper.make_node(
        "Split", ["x"], ["c0", "c1", "c2", "c3"], axis=1, num_outputs=4
    )

    def out(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    graph = helper.make_graph(
        [sizes, static_split, runtime_split, count_split],
        "main_graph",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [5, 3]),
            helper.make_tensor_value_info("runtime_sizes", TensorProto.INT64, [3]),
        ],
        [
            out("s0", [3, 3]),
            out("s1", [0, 3]),
            out("s2", [2, 3]),
            out("r0", [0, 3]),
            out("r1", [4, 3]),
            out("r2", [1, 3]),
            out("c0", [5, 1]),
            out("c1", [5, 1]),
            out("c2", [5, 1]),
            out("c3", [5, 0]),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "split_zero_size.onnx")

    x = np.arange(15, dtype=np.float32).reshape(5, 3)
    runtime_sizes = np.array([0, 4, 1], dtype=np.int64)
    outputs = ReferenceEvaluator(model).run(
        None, {"x": x, "runtime_sizes": runtime_sizes}
    )
    for name, value in zip(
        ["s0", "s1", "s2", "r0", "r1", "r2", "c0", "c1", "c2", "c3"], outputs
    ):
        print(name, value.shape, value.flatten().tolist())


if __name__ == "__main__":
    main()
