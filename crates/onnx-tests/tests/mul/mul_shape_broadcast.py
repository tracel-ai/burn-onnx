#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/mul/mul_shape_broadcast.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main():
    # Create a Shape-Shape operations
    input_1d = helper.make_tensor_value_info("input_1d", TensorProto.FLOAT, [3])
    input_4d = helper.make_tensor_value_info(
        "input_4d", TensorProto.FLOAT, [2, 30, 4, 5]
    )

    nodes = [
        helper.make_node("Shape", inputs=["input_1d"], outputs=["shape_1"]),
        helper.make_node("Shape", inputs=["input_4d"], outputs=["shape_4"]),
        helper.make_node("Mul", inputs=["shape_1", "shape_4"], outputs=["mul_lhs_bc"]),
        helper.make_node("Mul", inputs=["shape_4", "shape_1"], outputs=["mul_rhs_bc"]),
    ]

    outputs = [
        helper.make_tensor_value_info("mul_lhs_bc", TensorProto.INT64, [4]),
        helper.make_tensor_value_info("mul_rhs_bc", TensorProto.INT64, [4]),
    ]

    graph_def = helper.make_graph(
        nodes,
        "mul_shape_broadcast_test",
        [input_1d, input_4d],
        outputs,
    )
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model_def)

    feeds = {
        "input_1d": np.zeros(3, dtype=np.float32),
        "input_4d": np.zeros((2, 30, 4, 5), dtype=np.float32),
    }

    session = ReferenceEvaluator(model_def, verbose=0)
    mul_lhs_bc, mul_rhs_bc = session.run(None, feeds)

    print(f"mul_lhs_bc: {repr(mul_lhs_bc)}")
    print(f"mul_rhs_bc: {repr(mul_rhs_bc)}")

    np.testing.assert_array_equal(mul_lhs_bc, [6, 90, 12, 15])
    np.testing.assert_array_equal(mul_rhs_bc, [6, 90, 12, 15])

    onnx_name = "mul_shape_broadcast.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))


if __name__ == "__main__":
    main()
