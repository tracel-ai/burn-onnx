#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/and/and_shape_broadcast.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main():
    # 4 inputs are needed because ONNX And requires BOOL ops, so a Shape cannot feed it directly
    input_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [3])
    input_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [7])
    input_c = helper.make_tensor_value_info("input_c", TensorProto.FLOAT, [2, 3, 4, 5])
    input_d = helper.make_tensor_value_info("input_d", TensorProto.FLOAT, [9, 1, 4, 2])

    nodes = [
        helper.make_node("Shape", inputs=["input_a"], outputs=["shape_1a"]),
        helper.make_node("Shape", inputs=["input_b"], outputs=["shape_1b"]),
        helper.make_node("Shape", inputs=["input_c"], outputs=["shape_4a"]),
        helper.make_node("Shape", inputs=["input_d"], outputs=["shape_4b"]),
        # Shape(1) = [3 > 7] = [0]
        helper.make_node("Greater", inputs=["shape_1a", "shape_1b"], outputs=["greater_1"]),
        # Shape(4) = [2>9, 3>1, 4>4, 5>2] = [0, 1, 0, 1]
        helper.make_node("Greater", inputs=["shape_4a", "shape_4b"], outputs=["greater_4"]),
        # Shape(1) & Shape(4): lhs is the broadcast operand
        helper.make_node("And", inputs=["greater_1", "greater_4"], outputs=["and_lhs_bc"]),
        # Shape(4) & Shape(1): rhs is the broadcast operand
        helper.make_node("And", inputs=["greater_4", "greater_1"], outputs=["and_rhs_bc"]),
        helper.make_node("And", inputs=["greater_4", "greater_4"], outputs=["and_same"]),
    ]

    outputs = [
        helper.make_tensor_value_info("and_lhs_bc", TensorProto.BOOL, [4]),
        helper.make_tensor_value_info("and_rhs_bc", TensorProto.BOOL, [4]),
        helper.make_tensor_value_info("and_same", TensorProto.BOOL, [4]),
    ]

    graph_def = helper.make_graph(
        nodes,
        "and_shape_broadcast_test",
        [input_a, input_b, input_c, input_d],
        outputs,
    )
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model_def)

    # Shape discards the values, so only the declared ranks and extents matter.
    feeds = {
        "input_a": np.zeros(3, dtype=np.float32),
        "input_b": np.zeros(7, dtype=np.float32),
        "input_c": np.zeros((2, 3, 4, 5), dtype=np.float32),
        "input_d": np.zeros((9, 1, 4, 2), dtype=np.float32),
    }

    session = ReferenceEvaluator(model_def, verbose=0)
    and_lhs_bc, and_rhs_bc, and_same = session.run(None, feeds)

    print(f"and_lhs_bc: {repr(and_lhs_bc)}")
    print(f"and_rhs_bc: {repr(and_rhs_bc)}")
    print(f"and_same:   {repr(and_same)}")

    # greater_1 = [False], greater_4 = [False, True, False, True]
    np.testing.assert_array_equal(and_lhs_bc, [False, False, False, False])
    np.testing.assert_array_equal(and_rhs_bc, [False, False, False, False])
    np.testing.assert_array_equal(and_same, [False, True, False, True])

    onnx_name = "and_shape_broadcast.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))


if __name__ == "__main__":
    main()
