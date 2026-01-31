#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: dropout.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Constant", [], ["/dropout/Constant_output_0"],
        value=numpy_helper.from_array(np.array([0.25], dtype=np.float32).reshape([]), name="value"))
    node1 = helper.make_node(
        "Constant", [], ["/dropout/Constant_1_output_0"],
        value=numpy_helper.from_array(np.array([True], dtype=np.bool).reshape([]), name="value"))
    node2 = helper.make_node(
        "Dropout", ["input", "/dropout/Constant_output_0", "/dropout/Constant_1_output_0"], ["3", "/dropout/Dropout_output_1"])

    inp_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4, 10, 15])

    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [2, 4, 10, 15])

    graph = helper.make_graph(
        [node0, node1, node2],
        "main_graph",
        [inp_input],
        [out_n3],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "dropout.onnx")
    print(f"Finished exporting model to dropout.onnx")


if __name__ == "__main__":
    main()
