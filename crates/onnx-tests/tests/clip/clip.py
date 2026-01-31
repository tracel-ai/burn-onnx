#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: clip.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Constant", [], ["/Constant_output_0"],
        value=numpy_helper.from_array(np.array([0.30000001192092896], dtype=np.float32).reshape([]), name="value"))
    node1 = helper.make_node(
        "Clip", ["onnx::Clip_0", "/Constant_output_0", ""], ["5"])
    node2 = helper.make_node(
        "Constant", [], ["/Constant_1_output_0"],
        value=numpy_helper.from_array(np.array([0.5], dtype=np.float32).reshape([]), name="value"))
    node3 = helper.make_node(
        "Constant", [], ["/Constant_2_output_0"],
        value=numpy_helper.from_array(np.array([0.699999988079071], dtype=np.float32).reshape([]), name="value"))
    node4 = helper.make_node(
        "Clip", ["onnx::Clip_0", "/Constant_1_output_0", "/Constant_2_output_0"], ["10"])
    node5 = helper.make_node(
        "Constant", [], ["/Constant_3_output_0"],
        value=numpy_helper.from_array(np.array([0.800000011920929], dtype=np.float32).reshape([]), name="value"))
    node6 = helper.make_node(
        "Clip", ["onnx::Clip_0", "", "/Constant_3_output_0"], ["15"])

    inp_onnx__Clip_0 = helper.make_tensor_value_info("onnx::Clip_0", TensorProto.FLOAT, [6])

    out_n5 = helper.make_tensor_value_info("5", TensorProto.FLOAT, [6])
    out_n10 = helper.make_tensor_value_info("10", TensorProto.FLOAT, [6])
    out_n15 = helper.make_tensor_value_info("15", TensorProto.FLOAT, [6])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5, node6],
        "main_graph",
        [inp_onnx__Clip_0],
        [out_n5, out_n10, out_n15],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "clip.onnx")
    print(f"Finished exporting model to clip.onnx")


if __name__ == "__main__":
    main()
