#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Shape", ["x"], ["2"])

    inp_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])

    out_n2 = helper.make_tensor_value_info("2", TensorProto.INT64, [2])

    graph = helper.make_graph(
        [node0],
        "main_graph",
        [inp_x],
        [out_n2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "shape.onnx")
    print(f"Finished exporting model to shape.onnx")


if __name__ == "__main__":
    main()
