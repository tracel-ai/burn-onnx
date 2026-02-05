#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/max/max_broadcast.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Case 1: 3D vs 2D (broadcast rhs)
    node1 = helper.make_node("Max", ["x_3d", "y_2d"], ["result1"])
    # Case 2: 2D vs 3D (broadcast lhs)
    node2 = helper.make_node("Max", ["a_2d", "b_3d"], ["result2"])

    x_3d = helper.make_tensor_value_info("x_3d", TensorProto.FLOAT, [2, 3, 4])
    y_2d = helper.make_tensor_value_info("y_2d", TensorProto.FLOAT, [3, 4])
    a_2d = helper.make_tensor_value_info("a_2d", TensorProto.FLOAT, [3, 4])
    b_3d = helper.make_tensor_value_info("b_3d", TensorProto.FLOAT, [2, 3, 4])
    out1 = helper.make_tensor_value_info("result1", TensorProto.FLOAT, [2, 3, 4])
    out2 = helper.make_tensor_value_info("result2", TensorProto.FLOAT, [2, 3, 4])

    graph = helper.make_graph(
        [node1, node2], "main_graph", [x_3d, y_2d, a_2d, b_3d], [out1, out2]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.save(model, "max_broadcast.onnx")

    # Verify with reference evaluator
    ref = ReferenceEvaluator(model)
    x = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ],
        dtype=np.float32,
    )
    y = np.full([3, 4], 10.0, dtype=np.float32)
    r1, r2 = ref.run(None, {"x_3d": x, "y_2d": y, "a_2d": y, "b_3d": x})
    print(f"result1 (3D max 2D): {r1}")
    print(f"result2 (2D max 3D): {r2}")
    print("Finished exporting model to max_broadcast.onnx")


if __name__ == "__main__":
    main()
