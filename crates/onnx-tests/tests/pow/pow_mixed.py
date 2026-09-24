#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: pow_mixed.onnx
#
# Pow with mixed base and exponent types: float ^ int64 (with a negative base),
# int64 ^ float, and float ^ uint64.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    inputs = [
        helper.make_tensor_value_info("f", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("i", TensorProto.INT64, [4]),
        helper.make_tensor_value_info("fe", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("u", TensorProto.UINT64, [4]),
    ]
    outputs = [
        helper.make_tensor_value_info("float_int", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("int_float", TensorProto.INT64, [4]),
        helper.make_tensor_value_info("float_uint", TensorProto.FLOAT, [4]),
    ]
    nodes = [
        helper.make_node("Pow", ["f", "i"], ["float_int"]),
        helper.make_node("Pow", ["i", "fe"], ["int_float"]),
        helper.make_node("Pow", ["f", "u"], ["float_uint"]),
    ]
    graph = helper.make_graph(nodes, "pow_mixed", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "pow_mixed.onnx")

    feeds = {
        "f": np.array([-2.0, 1.5, 3.0, -1.0], dtype=np.float32),
        "i": np.array([3, 2, 4, 5], dtype=np.int64),
        "fe": np.array([2.0, 3.0, 0.5, 1.0], dtype=np.float32),
        "u": np.array([2, 3, 1, 4], dtype=np.uint64),
    }
    for name, value in zip(
        ["float_int", "int_float", "float_uint"], ReferenceEvaluator(model).run(None, feeds)
    ):
        print(f"{name}: {value.tolist()}")


if __name__ == "__main__":
    main()
