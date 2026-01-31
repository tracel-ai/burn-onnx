#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: equal.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Graph: Equal(x, ones_const) and Equal(Cast(k, float), 5.0_const)
    # x is float tensor [1,1,1,4], k is float64 scalar

    const_ones = helper.make_node(
        "Constant", [], ["ones"],
        value=numpy_helper.from_array(
            np.ones([1, 1, 1, 4], dtype=np.float32), name="ones"
        ),
    )
    equal1 = helper.make_node("Equal", ["x", "ones"], ["out1"])

    cast_k = helper.make_node("Cast", ["k"], ["k_float"], to=TensorProto.FLOAT)
    const_5 = helper.make_node(
        "Constant", [], ["five"],
        value=numpy_helper.from_array(np.float32(5.0), name="five"),
    )
    equal2 = helper.make_node("Equal", ["k_float", "five"], ["out2"])

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 4])
    k = helper.make_tensor_value_info("k", TensorProto.DOUBLE, [])
    out1 = helper.make_tensor_value_info("out1", TensorProto.BOOL, [1, 1, 1, 4])
    out2 = helper.make_tensor_value_info("out2", TensorProto.BOOL, [])

    graph = helper.make_graph(
        [const_ones, equal1, cast_k, const_5, equal2],
        "equal_test",
        [x, k],
        [out1, out2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx_name = "equal.onnx"
    onnx.save(model, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test
    test_x = np.ones([1, 1, 1, 4], dtype=np.float32)
    test_k = np.float64(2.0)
    session = ReferenceEvaluator(onnx_name)
    results = session.run(None, {"x": test_x, "k": test_k})
    print(f"Test input data: {test_x}, {test_k}")
    print(f"Test output data: {results}")


if __name__ == "__main__":
    main()
