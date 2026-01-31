#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/neg/neg.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Two Neg nodes: one for a float tensor, one for a float64 scalar
    neg1 = helper.make_node("Neg", ["x"], ["out1"])
    neg2 = helper.make_node("Neg", ["y"], ["out2"])

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.DOUBLE, [])
    out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 1, 1, 4])
    out2 = helper.make_tensor_value_info("out2", TensorProto.DOUBLE, [])

    graph = helper.make_graph([neg1, neg2], "neg_test", [x, y], [out1, out2])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx_name = "neg.onnx"
    onnx.save(model, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test
    test_x = np.array([[[[1.0, 4.0, 9.0, 25.0]]]]).astype(np.float32)
    test_y = np.float64(99.0)
    session = ReferenceEvaluator(onnx_name)
    out1, out2 = session.run(None, {"x": test_x, "y": test_y})
    print(f"Test input1: {test_x}, input2: {test_y}")
    print(f"Test output1 data: {out1}")
    print(f"Test output2 data: {out2}")


if __name__ == "__main__":
    main()
