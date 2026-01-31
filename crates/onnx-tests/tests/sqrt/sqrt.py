#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: sqrt.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Cast", ["onnx::Cast_1"], ["/Cast_output_0"], to=11)
    node1 = helper.make_node("Sqrt", ["onnx::Sqrt_0"], ["3"])
    node2 = helper.make_node("Sqrt", ["/Cast_output_0"], ["4"])

    inp_onnx__Sqrt_0 = helper.make_tensor_value_info(
        "onnx::Sqrt_0", TensorProto.FLOAT, [1, 1, 1, 4]
    )
    inp_onnx__Cast_1 = helper.make_tensor_value_info(
        "onnx::Cast_1", TensorProto.DOUBLE, []
    )

    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [1, 1, 1, 4])
    out_n4 = helper.make_tensor_value_info("4", TensorProto.DOUBLE, [])

    graph = helper.make_graph(
        [node0, node1, node2],
        "main_graph",
        [inp_onnx__Sqrt_0, inp_onnx__Cast_1],
        [out_n3, out_n4],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "sqrt.onnx")
    print(f"Finished exporting model to sqrt.onnx")


if __name__ == "__main__":
    main()
