#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: mod_scalar.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Cast", ["onnx::Cast_1"], ["/Cast_output_0"], to=1)
    node1 = helper.make_node("Mod", ["onnx::Mod_0", "/Cast_output_0"], ["3"], fmod=1)

    inp_onnx__Mod_0 = helper.make_tensor_value_info(
        "onnx::Mod_0", TensorProto.FLOAT, [1, 2, 3, 4]
    )
    inp_onnx__Cast_1 = helper.make_tensor_value_info(
        "onnx::Cast_1", TensorProto.DOUBLE, []
    )

    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [1, 2, 3, 4])

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_onnx__Mod_0, inp_onnx__Cast_1],
        [out_n3],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "mod_scalar.onnx")
    print(f"Finished exporting model to mod_scalar.onnx")


if __name__ == "__main__":
    main()
