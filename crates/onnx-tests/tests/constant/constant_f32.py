#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: constant_f32.onnx

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
            np.array([2.0], dtype=np.float32).reshape([]), name="value"
        ),
    )
    node1 = helper.make_node("Add", ["onnx::Add_0", "/Constant_output_0"], ["2"])

    inp_onnx__Add_0 = helper.make_tensor_value_info(
        "onnx::Add_0", TensorProto.FLOAT, [2, 3, 4]
    )

    out_n2 = helper.make_tensor_value_info("2", TensorProto.FLOAT, [2, 3, 4])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_onnx__Add_0],
        [out_n2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "constant_f32.onnx")
    print(f"Finished exporting model to constant_f32.onnx")


if __name__ == "__main__":
    main()
