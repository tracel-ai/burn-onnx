#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: div.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Cast", ["onnx::Cast_1"], ["/Cast_output_0"],
        to=1)
    node1 = helper.make_node(
        "Cast", ["onnx::Cast_2"], ["/Cast_1_output_0"],
        to=1)
    node2 = helper.make_node(
        "Div", ["/Cast_output_0", "/Cast_1_output_0"], ["/Div_output_0"])
    node3 = helper.make_node(
        "Div", ["onnx::Div_0", "/Div_output_0"], ["6"])

    inp_onnx__Div_0 = helper.make_tensor_value_info("onnx::Div_0", TensorProto.FLOAT, [1, 2, 3, 4])
    inp_onnx__Cast_1 = helper.make_tensor_value_info("onnx::Cast_1", TensorProto.DOUBLE, [])
    inp_onnx__Cast_2 = helper.make_tensor_value_info("onnx::Cast_2", TensorProto.DOUBLE, [])

    out_n6 = helper.make_tensor_value_info("6", TensorProto.FLOAT, [1, 2, 3, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3],
        "torch_jit",
        [inp_onnx__Div_0, inp_onnx__Cast_1, inp_onnx__Cast_2],
        [out_n6],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "div.onnx")
    print(f"Finished exporting model to div.onnx")


if __name__ == "__main__":
    main()
