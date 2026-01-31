#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: random_uniform.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "RandomUniform", [], ["1"],
        dtype=1,
        shape=[2, 3])


    out_n1 = helper.make_tensor_value_info("1", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [node0],
        "main_graph",
        [],
        [out_n1],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "random_uniform.onnx")
    print(f"Finished exporting model to random_uniform.onnx")


if __name__ == "__main__":
    main()
