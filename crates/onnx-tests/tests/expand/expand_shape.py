#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: expand_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Shape", ["shape_src"], ["/Shape_output_0"])
    node1 = helper.make_node(
        "Expand", ["inp", "/Shape_output_0"], ["3"])

    inp_inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [4, 1])
    inp_shape_src = helper.make_tensor_value_info("shape_src", TensorProto.FLOAT, [4, 4])

    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [4, 4])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_inp, inp_shape_src],
        [out_n3],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "expand_shape.onnx")
    print(f"Finished exporting model to expand_shape.onnx")


if __name__ == "__main__":
    main()
