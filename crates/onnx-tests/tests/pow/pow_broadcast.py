#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/pow/pow_broadcast.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Case 1: 3D base, 2D exponent (broadcast rhs)
    node1 = helper.make_node("Pow", ["base_3d", "exp_2d"], ["result1"])
    # Case 2: 2D base, 3D exponent (broadcast lhs)
    node2 = helper.make_node("Pow", ["base_2d", "exp_3d"], ["result2"])

    base_3d = helper.make_tensor_value_info("base_3d", TensorProto.FLOAT, [2, 3, 4])
    exp_2d = helper.make_tensor_value_info("exp_2d", TensorProto.FLOAT, [3, 4])
    base_2d = helper.make_tensor_value_info("base_2d", TensorProto.FLOAT, [3, 4])
    exp_3d = helper.make_tensor_value_info("exp_3d", TensorProto.FLOAT, [2, 3, 4])
    out1 = helper.make_tensor_value_info("result1", TensorProto.FLOAT, [2, 3, 4])
    out2 = helper.make_tensor_value_info("result2", TensorProto.FLOAT, [2, 3, 4])

    graph = helper.make_graph(
        [node1, node2],
        "main_graph",
        [base_3d, exp_2d, base_2d, exp_3d],
        [out1, out2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.save(model, "pow_broadcast.onnx")

    # Verify with reference evaluator
    ref = ReferenceEvaluator(model)
    base3d = np.full([2, 3, 4], 2.0, dtype=np.float32)
    exp2d = np.array(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        dtype=np.float32,
    )
    r1, r2 = ref.run(
        None,
        {"base_3d": base3d, "exp_2d": exp2d, "base_2d": exp2d, "exp_3d": base3d},
    )
    print(f"result1 (3D ** 2D): {r1}")
    print(f"result2 (2D ** 3D): {r2}")
    print("Finished exporting model to pow_broadcast.onnx")


if __name__ == "__main__":
    main()
