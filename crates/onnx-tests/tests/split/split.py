#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: split.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Constant", [], ["/Constant_output_0"],
        value=numpy_helper.from_array(np.array([2, 2, 1], dtype=np.int64).reshape([3]), name="value"))
    node1 = helper.make_node(
        "Split", ["tensor", "/Constant_output_0"], ["2", "3", "4"],
        axis=0)

    inp_tensor = helper.make_tensor_value_info("tensor", TensorProto.INT64, [5, 2])

    out_n2 = helper.make_tensor_value_info("2", TensorProto.INT64, [2, 2])
    out_n3 = helper.make_tensor_value_info("3", TensorProto.INT64, [2, 2])
    out_n4 = helper.make_tensor_value_info("4", TensorProto.INT64, [1, 2])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_tensor],
        [out_n2, out_n3, out_n4],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "split.onnx")
    print(f"Finished exporting model to split.onnx")


if __name__ == "__main__":
    main()
