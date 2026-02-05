#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: pow_int.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Pow", ["onnx::Pow_0", "onnx::Pow_0"], ["/Pow_output_0"])
    node1 = helper.make_node("Cast", ["onnx::Cast_1"], ["/Cast_output_0"], to=6)
    node2 = helper.make_node("Pow", ["/Pow_output_0", "/Cast_output_0"], ["4"])

    inp_onnx__Pow_0 = helper.make_tensor_value_info(
        "onnx::Pow_0", TensorProto.INT32, [1, 1, 1, 4]
    )
    inp_onnx__Cast_1 = helper.make_tensor_value_info(
        "onnx::Cast_1", TensorProto.INT64, []
    )

    out_n4 = helper.make_tensor_value_info("4", TensorProto.INT32, [1, 1, 1, 4])

    graph = helper.make_graph(
        [node0, node1, node2],
        "main_graph",
        [inp_onnx__Pow_0, inp_onnx__Cast_1],
        [out_n4],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "pow_int.onnx")
    print(f"Finished exporting model to pow_int.onnx")


if __name__ == "__main__":
    main()
