#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generate equal_scalar.onnx: Equal(tensor, scalar) and Equal(scalar, tensor)

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Test: Equal(tensor, scalar) and Equal(scalar, tensor)
    # x is a float tensor [1, 4], y is a float scalar

    # Equal(x, y) - tensor to scalar
    equal1 = helper.make_node("Equal", ["x", "y"], ["out1"])

    # Equal(y, x) - scalar to tensor
    equal2 = helper.make_node("Equal", ["y", "x"], ["out2"])

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])
    out1 = helper.make_tensor_value_info("out1", TensorProto.BOOL, [1, 4])
    out2 = helper.make_tensor_value_info("out2", TensorProto.BOOL, [1, 4])

    graph = helper.make_graph(
        [equal1, equal2],
        "equal_scalar_test",
        [x, y],
        [out1, out2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx_name = "equal_scalar.onnx"
    onnx.save(model, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test
    test_x = np.array([[1.0, 2.0, 3.0, 2.0]], dtype=np.float32)
    test_y = np.float32(2.0)
    session = ReferenceEvaluator(onnx_name)
    results = session.run(None, {"x": test_x, "y": test_y})
    print(f"Test input: x={test_x}, y={test_y}")
    print(f"  Equal(x, y) = {results[0]}")
    print(f"  Equal(y, x) = {results[1]}")


if __name__ == "__main__":
    main()
