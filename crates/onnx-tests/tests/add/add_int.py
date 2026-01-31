#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: add_int.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Add", ["onnx::Add_0", "onnx::Add_0"], ["/Add_output_0"])
    node1 = helper.make_node(
        "Constant",
        [],
        ["/Constant_output_0"],
        value=numpy_helper.from_array(
            np.array([5], dtype=np.int64).reshape([]), name="value"
        ),
    )
    node2 = helper.make_node(
        "Add", ["onnx::Add_1", "/Constant_output_0"], ["/Add_1_output_0"]
    )
    node3 = helper.make_node("Cast", ["/Add_1_output_0"], ["/Cast_output_0"], to=6)
    node4 = helper.make_node("Add", ["/Add_output_0", "/Cast_output_0"], ["6"])

    inp_onnx__Add_0 = helper.make_tensor_value_info(
        "onnx::Add_0", TensorProto.INT32, [1, 1, 1, 4]
    )
    inp_onnx__Add_1 = helper.make_tensor_value_info(
        "onnx::Add_1", TensorProto.INT64, []
    )

    out_n6 = helper.make_tensor_value_info("6", TensorProto.INT32, [1, 1, 1, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4],
        "torch_jit",
        [inp_onnx__Add_0, inp_onnx__Add_1],
        [out_n6],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "add_int.onnx")
    print(f"Finished exporting model to add_int.onnx")


if __name__ == "__main__":
    main()
