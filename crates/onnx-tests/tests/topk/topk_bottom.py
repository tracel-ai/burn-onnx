#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

# used to generate model: topk_bottom.onnx

import onnx
from onnx import helper, TensorProto

OPSET_VERSION = 16


def main():
    # constant tensor holding k=2 (used as second input)
    const_tensor = helper.make_tensor(
        name="value",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[2],
    )
    node0 = helper.make_node(
        "Constant",
        [],
        ["/Constant_output_0"],
        value=const_tensor,
    )
    node1 = helper.make_node(
        "TopK",
        ["onnx::TopK_0", "/Constant_output_0"],
        ["4", "5"],
        axis=1,
        largest=0,  # bottom-k
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

    onnx.save(model, "topk_bottom.onnx")
    print(f"Finished exporting model to topk_bottom.onnx")


if __name__ == "__main__":
    main()
