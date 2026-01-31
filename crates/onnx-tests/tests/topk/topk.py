#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: topk.onnx

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
        "TopK",
        ["onnx::TopK_0", "/Constant_output_0"],
        ["4", "5"],
        axis=1,
        largest=1,
        sorted=1,
    )

    inp_onnx__TopK_0 = helper.make_tensor_value_info(
        "onnx::TopK_0", TensorProto.FLOAT, [3, 5]
    )

    out_n4 = helper.make_tensor_value_info("4", TensorProto.FLOAT, [3, 2])
    out_n5 = helper.make_tensor_value_info("5", TensorProto.INT64, [3, 2])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_onnx__TopK_0],
        [out_n4, out_n5],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "topk.onnx")
    print(f"Finished exporting model to topk.onnx")


if __name__ == "__main__":
    main()
