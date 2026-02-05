#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: sub.onnx

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
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape([1, 1, 1, 4]),
            name="value",
        ),
    )
    node1 = helper.make_node(
        "Sub", ["onnx::Sub_0", "/Constant_output_0"], ["/Sub_output_0"]
    )
    node2 = helper.make_node("Cast", ["onnx::Cast_1"], ["/Cast_output_0"], to=1)
    node3 = helper.make_node(
        "Constant",
        [],
        ["/Constant_1_output_0"],
        value=numpy_helper.from_array(
            np.array([9.0], dtype=np.float32).reshape([]), name="value"
        ),
    )
    node4 = helper.make_node(
        "Sub", ["/Cast_output_0", "/Constant_1_output_0"], ["/Sub_1_output_0"]
    )
    node5 = helper.make_node(
        "Sub", ["/Sub_output_0", "/Sub_1_output_0"], ["/Sub_2_output_0"]
    )
    node6 = helper.make_node("Sub", ["/Sub_1_output_0", "/Sub_2_output_0"], ["8"])

    inp_onnx__Sub_0 = helper.make_tensor_value_info(
        "onnx::Sub_0", TensorProto.FLOAT, [1, 2, 3, 4]
    )
    inp_onnx__Cast_1 = helper.make_tensor_value_info(
        "onnx::Cast_1", TensorProto.DOUBLE, []
    )

    out_n8 = helper.make_tensor_value_info("8", TensorProto.FLOAT, [1, 2, 3, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5, node6],
        "main_graph",
        [inp_onnx__Sub_0, inp_onnx__Cast_1],
        [out_n8],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "sub.onnx")
    print(f"Finished exporting model to sub.onnx")


if __name__ == "__main__":
    main()
