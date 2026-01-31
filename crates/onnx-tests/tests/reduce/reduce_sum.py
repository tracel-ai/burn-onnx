#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: reduce_sum.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "ReduceSum", ["onnx::ReduceSum_0"], ["2"],
        keepdims=0)
    node1 = helper.make_node(
        "ReduceSum", ["onnx::ReduceSum_0", ""], ["3"],
        keepdims=0)
    node2 = helper.make_node(
        "Constant", [], ["onnx::ReduceSum_4"],
        value=numpy_helper.from_array(np.array([1], dtype=np.int64).reshape([1]), name="value"))
    node3 = helper.make_node(
        "ReduceSum", ["onnx::ReduceSum_0", "onnx::ReduceSum_4"], ["5"],
        keepdims=1)
    node4 = helper.make_node(
        "Constant", [], ["onnx::ReduceSum_6"],
        value=numpy_helper.from_array(np.array([-1], dtype=np.int64).reshape([1]), name="value"))
    node5 = helper.make_node(
        "ReduceSum", ["onnx::ReduceSum_0", "onnx::ReduceSum_6"], ["7"],
        keepdims=1)
    node6 = helper.make_node(
        "Constant", [], ["onnx::ReduceSum_8"],
        value=numpy_helper.from_array(np.array([0], dtype=np.int64).reshape([1]), name="value"))
    node7 = helper.make_node(
        "ReduceSum", ["onnx::ReduceSum_0", "onnx::ReduceSum_8"], ["9"],
        keepdims=0)
    node8 = helper.make_node(
        "Constant", [], ["onnx::ReduceSum_10"],
        value=numpy_helper.from_array(np.array([0, 2], dtype=np.int64).reshape([2]), name="value"))
    node9 = helper.make_node(
        "ReduceSum", ["onnx::ReduceSum_0", "onnx::ReduceSum_10"], ["11"],
        keepdims=0)

    inp_onnx__ReduceSum_0 = helper.make_tensor_value_info("onnx::ReduceSum_0", TensorProto.FLOAT, [1, 1, 2, 4])

    out_n2 = helper.make_tensor_value_info("2", TensorProto.FLOAT, [])
    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [])
    out_n5 = helper.make_tensor_value_info("5", TensorProto.FLOAT, [1, 1, 2, 4])
    out_n7 = helper.make_tensor_value_info("7", TensorProto.FLOAT, [1, 1, 2, 1])
    out_n9 = helper.make_tensor_value_info("9", TensorProto.FLOAT, [1, 2, 4])
    out_n11 = helper.make_tensor_value_info("11", TensorProto.FLOAT, [1, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5, node6, node7, node8, node9],
        "main_graph",
        [inp_onnx__ReduceSum_0],
        [out_n2, out_n3, out_n5, out_n7, out_n9, out_n11],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "reduce_sum.onnx")
    print(f"Finished exporting model to reduce_sum.onnx")


if __name__ == "__main__":
    main()
