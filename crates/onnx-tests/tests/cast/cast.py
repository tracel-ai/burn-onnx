#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: cast.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Cast", ["onnx::Cast_0"], ["4"],
        to=9)
    node1 = helper.make_node(
        "Cast", ["onnx::Cast_0"], ["5"],
        to=6)
    node2 = helper.make_node(
        "Cast", ["onnx::Cast_0"], ["6"],
        to=1)
    node3 = helper.make_node(
        "Cast", ["onnx::Cast_1"], ["7"],
        to=9)
    node4 = helper.make_node(
        "Cast", ["onnx::Cast_1"], ["8"],
        to=6)
    node5 = helper.make_node(
        "Cast", ["onnx::Cast_1"], ["9"],
        to=1)
    node6 = helper.make_node(
        "Cast", ["onnx::Cast_2"], ["10"],
        to=9)
    node7 = helper.make_node(
        "Cast", ["onnx::Cast_2"], ["11"],
        to=6)
    node8 = helper.make_node(
        "Cast", ["onnx::Cast_2"], ["12"],
        to=1)
    node9 = helper.make_node(
        "Cast", ["onnx::Cast_3"], ["13"],
        to=6)

    inp_onnx__Cast_0 = helper.make_tensor_value_info("onnx::Cast_0", TensorProto.BOOL, [2, 1])
    inp_onnx__Cast_1 = helper.make_tensor_value_info("onnx::Cast_1", TensorProto.INT32, [2, 1])
    inp_onnx__Cast_2 = helper.make_tensor_value_info("onnx::Cast_2", TensorProto.FLOAT, [2, 1])
    inp_onnx__Cast_3 = helper.make_tensor_value_info("onnx::Cast_3", TensorProto.FLOAT, [])

    out_n4 = helper.make_tensor_value_info("4", TensorProto.BOOL, [2, 1])
    out_n5 = helper.make_tensor_value_info("5", TensorProto.INT32, [2, 1])
    out_n6 = helper.make_tensor_value_info("6", TensorProto.FLOAT, [2, 1])
    out_n7 = helper.make_tensor_value_info("7", TensorProto.BOOL, [2, 1])
    out_n8 = helper.make_tensor_value_info("8", TensorProto.INT32, [2, 1])
    out_n9 = helper.make_tensor_value_info("9", TensorProto.FLOAT, [2, 1])
    out_n10 = helper.make_tensor_value_info("10", TensorProto.BOOL, [2, 1])
    out_n11 = helper.make_tensor_value_info("11", TensorProto.INT32, [2, 1])
    out_n12 = helper.make_tensor_value_info("12", TensorProto.FLOAT, [2, 1])
    out_n13 = helper.make_tensor_value_info("13", TensorProto.INT32, [])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5, node6, node7, node8, node9],
        "main_graph",
        [inp_onnx__Cast_0, inp_onnx__Cast_1, inp_onnx__Cast_2, inp_onnx__Cast_3],
        [out_n4, out_n5, out_n6, out_n7, out_n8, out_n9, out_n10, out_n11, out_n12, out_n13],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "cast.onnx")
    print(f"Finished exporting model to cast.onnx")


if __name__ == "__main__":
    main()
