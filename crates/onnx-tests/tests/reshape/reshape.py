#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: reshape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Constant", [], ["/Constant_output_0"],
        value=numpy_helper.from_array(np.array([2, 2], dtype=np.int64).reshape([2]), name="value"))
    node1 = helper.make_node(
        "Reshape", ["onnx::Reshape_0", "/Constant_output_0"], ["/Reshape_output_0"],
        allowzero=0)
    node2 = helper.make_node(
        "Constant", [], ["/Constant_1_output_0"],
        value=numpy_helper.from_array(np.array([1, -1], dtype=np.int64).reshape([2]), name="value"))
    node3 = helper.make_node(
        "Reshape", ["/Reshape_output_0", "/Constant_1_output_0"], ["4"],
        allowzero=0)

    inp_onnx__Reshape_0 = helper.make_tensor_value_info("onnx::Reshape_0", TensorProto.FLOAT, [4])

    out_n4 = helper.make_tensor_value_info("4", TensorProto.FLOAT, [1, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3],
        "torch_jit",
        [inp_onnx__Reshape_0],
        [out_n4],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "reshape.onnx")
    print(f"Finished exporting model to reshape.onnx")


if __name__ == "__main__":
    main()
