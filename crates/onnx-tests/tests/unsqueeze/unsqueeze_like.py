#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: unsqueeze_like.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Constant",
        [],
        ["/Constant_output_0"],
        value=numpy_helper.from_array(
            np.array([3], dtype=np.int64).reshape([1]), name="value"
        ),
    )
    node1 = helper.make_node(
        "Unsqueeze", ["onnx::Unsqueeze_0", "/Constant_output_0"], ["3"]
    )
    node2 = helper.make_node("Cast", ["onnx::Cast_1"], ["/Cast_output_0"], to=11)
    node3 = helper.make_node(
        "Constant",
        [],
        ["/Constant_1_output_0"],
        value=numpy_helper.from_array(
            np.array([0], dtype=np.int64).reshape([1]), name="value"
        ),
    )
    node4 = helper.make_node(
        "Unsqueeze", ["/Cast_output_0", "/Constant_1_output_0"], ["6"]
    )

    inp_onnx__Unsqueeze_0 = helper.make_tensor_value_info(
        "onnx::Unsqueeze_0", TensorProto.FLOAT, [3, 4, 5]
    )
    inp_onnx__Cast_1 = helper.make_tensor_value_info(
        "onnx::Cast_1", TensorProto.DOUBLE, []
    )

    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [3, 4, 5, 1])
    out_n6 = helper.make_tensor_value_info("6", TensorProto.DOUBLE, [1])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4],
        "main_graph",
        [inp_onnx__Unsqueeze_0, inp_onnx__Cast_1],
        [out_n3, out_n6],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "unsqueeze_like.onnx")
    print(f"Finished exporting model to unsqueeze_like.onnx")


if __name__ == "__main__":
    main()
