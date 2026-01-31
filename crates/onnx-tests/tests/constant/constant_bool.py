#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: constant_bool.onnx

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
            np.array([True], dtype=np.bool).reshape([]), name="value"
        ),
    )
    node1 = helper.make_node("Or", ["onnx::Or_0", "/Constant_output_0"], ["2"])

    inp_onnx__Or_0 = helper.make_tensor_value_info("onnx::Or_0", TensorProto.BOOL, [])

    out_n2 = helper.make_tensor_value_info("2", TensorProto.BOOL, [])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_onnx__Or_0],
        [out_n2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "constant_bool.onnx")
    print(f"Finished exporting model to constant_bool.onnx")


if __name__ == "__main__":
    main()
