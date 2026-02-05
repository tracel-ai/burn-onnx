#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: squeeze.onnx

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
            np.array([2], dtype=np.int64).reshape([1]), name="value"
        ),
    )
    node1 = helper.make_node(
        "Squeeze", ["onnx::Squeeze_0", "/Constant_output_0"], ["2"]
    )

    inp_onnx__Squeeze_0 = helper.make_tensor_value_info(
        "onnx::Squeeze_0", TensorProto.FLOAT, [3, 4, 1, 5]
    )

    out_n2 = helper.make_tensor_value_info("2", TensorProto.FLOAT, [3, 4, 5])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_onnx__Squeeze_0],
        [out_n2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "squeeze.onnx")
    print(f"Finished exporting model to squeeze.onnx")


if __name__ == "__main__":
    main()
