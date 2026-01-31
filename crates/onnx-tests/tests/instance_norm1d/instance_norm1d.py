#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: instance_norm1d.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Constant", [], ["/norm1/Constant_output_0"],
        value=numpy_helper.from_array(np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape([3]), name="value"))
    node1 = helper.make_node(
        "Constant", [], ["/norm1/Constant_1_output_0"],
        value=numpy_helper.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape([3]), name="value"))
    node2 = helper.make_node(
        "InstanceNormalization", ["input", "/norm1/Constant_output_0", "/norm1/Constant_1_output_0"], ["3"],
        epsilon=9.999999747378752e-06)

    inp_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])

    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [2, 3, 4])

    graph = helper.make_graph(
        [node0, node1, node2],
        "main_graph",
        [inp_input],
        [out_n3],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "instance_norm1d.onnx")
    print(f"Finished exporting model to instance_norm1d.onnx")


if __name__ == "__main__":
    main()
