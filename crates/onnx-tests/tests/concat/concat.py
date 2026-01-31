#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: concat.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Concat", ["onnx::Concat_0", "onnx::Concat_0"], ["/Concat_output_0"], axis=1
    )
    node1 = helper.make_node(
        "Concat",
        [
            "onnx::Concat_0",
            "/Concat_output_0",
            "/Concat_output_0",
            "/Concat_output_0",
            "/Concat_output_0",
        ],
        ["2"],
        axis=1,
    )

    inp_onnx__Concat_0 = helper.make_tensor_value_info(
        "onnx::Concat_0", TensorProto.FLOAT, [1, 2, 3, 5]
    )

    out_n2 = helper.make_tensor_value_info("2", TensorProto.FLOAT, [1, 18, 3, 5])

    graph = helper.make_graph(
        [node0, node1],
        "torch_jit",
        [inp_onnx__Concat_0],
        [out_n2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "concat.onnx")
    print(f"Finished exporting model to concat.onnx")


if __name__ == "__main__":
    main()
