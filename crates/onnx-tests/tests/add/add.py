#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: add.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Constant", [], ["/Constant_output_0"],
        value=numpy_helper.from_array(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape([1, 1, 1, 4]), name="value"))
    node1 = helper.make_node(
        "Add", ["onnx::Add_0", "/Constant_output_0"], ["/Add_output_0"])
    node2 = helper.make_node(
        "Cast", ["onnx::Cast_1"], ["/Cast_output_0"],
        to=1)
    node3 = helper.make_node(
        "Constant", [], ["/Constant_1_output_0"],
        value=numpy_helper.from_array(np.array([5.0], dtype=np.float32).reshape([]), name="value"))
    node4 = helper.make_node(
        "Add", ["/Cast_output_0", "/Constant_1_output_0"], ["/Add_1_output_0"])
    node5 = helper.make_node(
        "Add", ["/Add_output_0", "/Add_1_output_0"], ["7"])

    inp_onnx__Add_0 = helper.make_tensor_value_info("onnx::Add_0", TensorProto.FLOAT, [1, 2, 3, 4])
    inp_onnx__Cast_1 = helper.make_tensor_value_info("onnx::Cast_1", TensorProto.DOUBLE, [])

    out_n7 = helper.make_tensor_value_info("7", TensorProto.FLOAT, [1, 2, 3, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5],
        "torch_jit",
        [inp_onnx__Add_0, inp_onnx__Cast_1],
        [out_n7],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "add.onnx")
    print(f"Finished exporting model to add.onnx")


if __name__ == "__main__":
    main()
