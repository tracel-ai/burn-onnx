#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: flatten.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Flatten", ["onnx::Flatten_0"], ["1"],
        axis=1)

    inp_onnx__Flatten_0 = helper.make_tensor_value_info("onnx::Flatten_0", TensorProto.FLOAT, [1, 5, 15])

    out_n1 = helper.make_tensor_value_info("1", TensorProto.FLOAT, [1, 75])

    graph = helper.make_graph(
        [node0],
        "torch_jit",
        [inp_onnx__Flatten_0],
        [out_n1],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "flatten.onnx")
    print(f"Finished exporting model to flatten.onnx")


if __name__ == "__main__":
    main()
