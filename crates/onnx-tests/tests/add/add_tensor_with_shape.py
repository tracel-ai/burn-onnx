#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/add/add_tensor_with_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Graph: Add(input_1d * 2, Shape(input_tensor))
    # Tests adding a tensor with a shape output

    input_tensor = helper.make_tensor_value_info(
        "input_tensor", TensorProto.FLOAT, [2, 3, 4]
    )
    input_1d = helper.make_tensor_value_info("input_1d", TensorProto.INT64, [3])

    shape_node = helper.make_node("Shape", ["input_tensor"], ["shape"])

    const_multiplier = helper.make_node(
        "Constant",
        [],
        ["multiplier"],
        value=helper.make_tensor("const_mult", TensorProto.INT64, [], [2]),
    )

    mul_node = helper.make_node("Mul", ["input_1d", "multiplier"], ["mul_result"])
    add_node = helper.make_node("Add", ["mul_result", "shape"], ["tensor_plus_shape"])

    output = helper.make_tensor_value_info("tensor_plus_shape", TensorProto.INT64, [3])

    graph = helper.make_graph(
        [shape_node, const_multiplier, mul_node, add_node],
        "tensor_with_shape_add_test",
        [input_tensor, input_1d],
        [output],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx_name = "add_tensor_with_shape.onnx"
    onnx.save(model, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test
    test_input = np.random.randn(2, 3, 4).astype(np.float32)
    test_1d = np.array([2, 3, 4], dtype=np.int64)

    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test 1d tensor: {test_1d}")

    session = ReferenceEvaluator(onnx_name)
    outputs = session.run(None, {"input_tensor": test_input, "input_1d": test_1d})
    print(f"\nTest output tensor_plus_shape: {repr(outputs[0])}")


if __name__ == "__main__":
    main()
