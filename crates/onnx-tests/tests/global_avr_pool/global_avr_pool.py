#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: global_avr_pool.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("GlobalAveragePool", ["onnx::GlobalAveragePool_0"], ["2"])
    node1 = helper.make_node("GlobalAveragePool", ["input"], ["3"])

    inp_onnx__GlobalAveragePool_0 = helper.make_tensor_value_info(
        "onnx::GlobalAveragePool_0", TensorProto.FLOAT, [2, 4, 10]
    )
    inp_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [3, 10, 3, 15]
    )

    out_n2 = helper.make_tensor_value_info("2", TensorProto.FLOAT, [2, 4, 1])
    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [3, 10, 1, 1])

    graph = helper.make_graph(
        [node0, node1],
        "torch_jit",
        [inp_onnx__GlobalAveragePool_0, inp_input],
        [out_n2, out_n3],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "global_avr_pool.onnx")
    print(f"Finished exporting model to global_avr_pool.onnx")


if __name__ == "__main__":
    main()
