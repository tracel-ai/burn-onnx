#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: reduce_mean.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "ReduceMean", ["onnx::ReduceMean_0"], ["1"],
        keepdims=0)
    node1 = helper.make_node(
        "ReduceMean", ["onnx::ReduceMean_0"], ["2"],
        keepdims=0)
    node2 = helper.make_node(
        "ReduceMean", ["onnx::ReduceMean_0"], ["3"],
        axes=[1],
        keepdims=1)
    node3 = helper.make_node(
        "ReduceMean", ["onnx::ReduceMean_0"], ["4"],
        axes=[-1],
        keepdims=1)
    node4 = helper.make_node(
        "ReduceMean", ["onnx::ReduceMean_0"], ["5"],
        axes=[0],
        keepdims=0)
    node5 = helper.make_node(
        "ReduceMean", ["onnx::ReduceMean_0"], ["6"],
        axes=[0, 2],
        keepdims=0)

    inp_onnx__ReduceMean_0 = helper.make_tensor_value_info("onnx::ReduceMean_0", TensorProto.FLOAT, [1, 1, 2, 4])

    out_n1 = helper.make_tensor_value_info("1", TensorProto.FLOAT, [])
    out_n2 = helper.make_tensor_value_info("2", TensorProto.FLOAT, [])
    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [1, 1, 2, 4])
    out_n4 = helper.make_tensor_value_info("4", TensorProto.FLOAT, [1, 1, 2, 1])
    out_n5 = helper.make_tensor_value_info("5", TensorProto.FLOAT, [1, 2, 4])
    out_n6 = helper.make_tensor_value_info("6", TensorProto.FLOAT, [1, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5],
        "main_graph",
        [inp_onnx__ReduceMean_0],
        [out_n1, out_n2, out_n3, out_n4, out_n5, out_n6],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "reduce_mean.onnx")
    print(f"Finished exporting model to reduce_mean.onnx")


if __name__ == "__main__":
    main()
