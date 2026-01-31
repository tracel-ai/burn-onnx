#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: trilu_lower.onnx

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
            np.array([0], dtype=np.int64).reshape([]), name="value"
        ),
    )
    node1 = helper.make_node(
        "Trilu", ["onnx::Trilu_0", "/Constant_output_0"], ["2"], upper=0
    )

    inp_onnx__Trilu_0 = helper.make_tensor_value_info(
        "onnx::Trilu_0", TensorProto.FLOAT, [2, 4, 4]
    )

    out_n2 = helper.make_tensor_value_info("2", TensorProto.FLOAT, [2, 4, 4])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_onnx__Trilu_0],
        [out_n2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "trilu_lower.onnx")
    print(f"Finished exporting model to trilu_lower.onnx")


if __name__ == "__main__":
    main()
