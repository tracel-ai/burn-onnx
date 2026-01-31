#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/where_op/where_static_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Where(condition, x_const, y_const) with static constant shapes [2,2]

    const_x = helper.make_node(
        "Constant",
        [],
        ["x"],
        value=numpy_helper.from_array(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="x"
        ),
    )
    const_y = helper.make_node(
        "Constant",
        [],
        ["y"],
        value=numpy_helper.from_array(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), name="y"
        ),
    )
    where_node = helper.make_node("Where", ["condition", "x", "y"], ["output"])

    condition = helper.make_tensor_value_info(
        "condition", TensorProto.BOOL, [None, None]
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])

    graph = helper.make_graph(
        [const_x, const_y, where_node],
        "where_static_shape_test",
        [condition],
        [output],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx_name = "where_static_shape.onnx"
    onnx.save(model, onnx_name)
    print(f"Model exported to {onnx_name}")

    # Test
    cond = np.array([[True, False], [False, True]])
    session = ReferenceEvaluator(onnx_name)
    outputs = session.run(None, {"condition": cond})
    print(f"Condition shape: {cond.shape}")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output:\n{outputs[0]}")

    expected = np.array([[1.0, 6.0], [7.0, 4.0]], dtype=np.float32)
    assert np.allclose(outputs[0], expected), f"Expected {expected}, got {outputs[0]}"
    print("Test passed!")


if __name__ == "__main__":
    main()
