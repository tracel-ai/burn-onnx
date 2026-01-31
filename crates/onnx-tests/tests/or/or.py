#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: or.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Cast", ["onnx::Cast_0"], ["/Cast_output_0"], to=9)
    node1 = helper.make_node("Cast", ["onnx::Cast_1"], ["/Cast_1_output_0"], to=9)
    node2 = helper.make_node("Or", ["/Cast_output_0", "/Cast_1_output_0"], ["4"])

    inp_onnx__Cast_0 = helper.make_tensor_value_info(
        "onnx::Cast_0", TensorProto.BOOL, [1, 1, 1, 4]
    )
    inp_onnx__Cast_1 = helper.make_tensor_value_info(
        "onnx::Cast_1", TensorProto.BOOL, [1, 1, 1, 4]
    )

    out_n4 = helper.make_tensor_value_info("4", TensorProto.BOOL, [1, 1, 1, 4])

    graph = helper.make_graph(
        [node0, node1, node2],
        "main_graph",
        [inp_onnx__Cast_0, inp_onnx__Cast_1],
        [out_n4],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "or.onnx")
    print(f"Finished exporting model to or.onnx")


if __name__ == "__main__":
    main()
