#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: is_inf.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Cast", ["onnx::Cast_0"], ["/Cast_output_0"],
        to=11)
    node1 = helper.make_node(
        "IsInf", ["/Cast_output_0"], ["2"])

    inp_onnx__Cast_0 = helper.make_tensor_value_info("onnx::Cast_0", TensorProto.FLOAT, [4, 4])

    out_n2 = helper.make_tensor_value_info("2", TensorProto.BOOL, [4, 4])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_onnx__Cast_0],
        [out_n2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "is_inf.onnx")
    print(f"Finished exporting model to is_inf.onnx")


if __name__ == "__main__":
    main()
