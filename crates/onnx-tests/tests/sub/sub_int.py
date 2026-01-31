#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: sub_int.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Sub", ["onnx::Sub_0", "onnx::Sub_0"], ["/Sub_output_0"])
    node1 = helper.make_node(
        "Constant",
        [],
        ["/Constant_output_0"],
        value=numpy_helper.from_array(
            np.array([9], dtype=np.int64).reshape([]), name="value"
        ),
    )
    node2 = helper.make_node(
        "Sub", ["onnx::Sub_1", "/Constant_output_0"], ["/Sub_1_output_0"]
    )
    node3 = helper.make_node(
        "Sub", ["/Sub_output_0", "/Sub_1_output_0"], ["/Sub_2_output_0"]
    )
    node4 = helper.make_node("Sub", ["/Sub_1_output_0", "/Sub_2_output_0"], ["6"])

    inp_onnx__Sub_0 = helper.make_tensor_value_info(
        "onnx::Sub_0", TensorProto.INT64, [1, 1, 1, 4]
    )
    inp_onnx__Sub_1 = helper.make_tensor_value_info(
        "onnx::Sub_1", TensorProto.INT64, []
    )

    out_n6 = helper.make_tensor_value_info("6", TensorProto.INT64, [1, 1, 1, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4],
        "main_graph",
        [inp_onnx__Sub_0, inp_onnx__Sub_1],
        [out_n6],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "sub_int.onnx")
    print(f"Finished exporting model to sub_int.onnx")


if __name__ == "__main__":
    main()
