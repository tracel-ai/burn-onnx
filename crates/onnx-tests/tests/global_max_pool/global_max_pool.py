#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: global_max_pool.onnx

import onnx
from onnx import TensorProto, helper

OPSET_VERSION = 22


def main():
    node0 = helper.make_node("GlobalMaxPool", ["onnx::GlobalMaxPool_0"], ["2"])
    node1 = helper.make_node("GlobalMaxPool", ["input"], ["3"])

    inp_onnx__GlobalMaxPool_0 = helper.make_tensor_value_info(
        "onnx::GlobalMaxPool_0", TensorProto.FLOAT, [2, 4, 10]
    )
    inp_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [3, 10, 3, 15]
    )

    out_n2 = helper.make_tensor_value_info("2", TensorProto.FLOAT, [2, 4, 1])
    out_n3 = helper.make_tensor_value_info("3", TensorProto.FLOAT, [3, 10, 1, 1])

    graph = helper.make_graph(
        [node0, node1],
        "torch_jit",
        [inp_onnx__GlobalMaxPool_0, inp_input],
        [out_n2, out_n3],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "global_max_pool.onnx")
    print("Finished exporting model to global_max_pool.onnx")


if __name__ == "__main__":
    main()
