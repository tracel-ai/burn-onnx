#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: constant_of_shape_full_like.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Shape", ["input"], ["/Shape_output_0"])
    node1 = helper.make_node(
        "ConstantOfShape", ["/Shape_output_0"], ["output_float"],
        value=numpy_helper.from_array(np.array([3.0], dtype=np.float32).reshape([1]), name="value"))
    node2 = helper.make_node(
        "Shape", ["input"], ["/Shape_1_output_0"])
    node3 = helper.make_node(
        "ConstantOfShape", ["/Shape_1_output_0"], ["output_int"],
        value=numpy_helper.from_array(np.array([5], dtype=np.int32).reshape([1]), name="value"))
    node4 = helper.make_node(
        "Shape", ["input"], ["/Shape_2_output_0"])
    node5 = helper.make_node(
        "ConstantOfShape", ["/Shape_2_output_0"], ["output_bool"],
        value=numpy_helper.from_array(np.array([True], dtype=np.bool).reshape([1]), name="value"))

    inp_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None, None])

    out_output_float = helper.make_tensor_value_info("output_float", TensorProto.FLOAT, [None, None, None])
    out_output_int = helper.make_tensor_value_info("output_int", TensorProto.INT32, [None, None, None])
    out_output_bool = helper.make_tensor_value_info("output_bool", TensorProto.BOOL, [None, None, None])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5],
        "main_graph",
        [inp_input],
        [out_output_float, out_output_int, out_output_bool],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "constant_of_shape_full_like.onnx")
    print(f"Finished exporting model to constant_of_shape_full_like.onnx")


if __name__ == "__main__":
    main()
