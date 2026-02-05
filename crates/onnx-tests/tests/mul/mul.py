#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: mul.onnx

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
            np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32).reshape([1, 1, 1, 4]),
            name="value",
        ),
    )
    node1 = helper.make_node(
        "Mul", ["onnx::Mul_0", "/Constant_output_0"], ["/Mul_output_0"]
    )
    node2 = helper.make_node("Cast", ["onnx::Cast_1"], ["/Cast_output_0"], to=1)
    node3 = helper.make_node(
        "Constant",
        [],
        ["/Constant_1_output_0"],
        value=numpy_helper.from_array(
            np.array([7.0], dtype=np.float32).reshape([]), name="value"
        ),
    )
    node4 = helper.make_node(
        "Mul", ["/Cast_output_0", "/Constant_1_output_0"], ["/Mul_1_output_0"]
    )
    node5 = helper.make_node("Mul", ["/Mul_output_0", "/Mul_1_output_0"], ["7"])

    inp_onnx__Mul_0 = helper.make_tensor_value_info(
        "onnx::Mul_0", TensorProto.FLOAT, [1, 2, 3, 4]
    )
    inp_onnx__Cast_1 = helper.make_tensor_value_info(
        "onnx::Cast_1", TensorProto.DOUBLE, []
    )

    out_n7 = helper.make_tensor_value_info("7", TensorProto.FLOAT, [1, 2, 3, 4])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5],
        "torch_jit",
        [inp_onnx__Mul_0, inp_onnx__Cast_1],
        [out_n7],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "mul.onnx")
    print(f"Finished exporting model to mul.onnx")


if __name__ == "__main__":
    main()
