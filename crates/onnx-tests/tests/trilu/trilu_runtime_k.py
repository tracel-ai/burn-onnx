#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: trilu_runtime_k.onnx

import onnx
from onnx import helper, TensorProto

OPSET_VERSION = 16


def main():
    # The diagonal offset `k` is a graph input, so it is only known at runtime
    node0 = helper.make_node("Trilu", ["x", "k"], ["y"], upper=0)

    inp_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 3])
    inp_k = helper.make_tensor_value_info("k", TensorProto.INT64, [])

    out_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 3])

    graph = helper.make_graph(
        [node0],
        "main_graph",
        [inp_x, inp_k],
        [out_y],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "trilu_runtime_k.onnx")
    print(f"Finished exporting model to trilu_runtime_k.onnx")


if __name__ == "__main__":
    main()
