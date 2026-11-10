#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: scatter_elements_3d.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2, 4])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2, 2, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 4])

    # Middle axis on a rank-3 tensor, so the non-axis coordinate columns exercise
    # both a non-unit outer stride and the unit inner stride.
    node = helper.make_node(
        "ScatterElements",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        axis=1,
        reduction="max",
    )

    graph = helper.make_graph(
        [node], "scatter_elements_3d_graph", [data, indices, updates], [output]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_elements_3d.onnx")

    np.random.seed(42)
    test_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    # Mixes negative and positive indices along the middle axis.
    # No column repeats a target row, so the fixture does not depend on how
    # duplicate indices fold.
    test_indices = np.array(
        [[[0, -1, 1, -2], [2, 1, -3, 0]], [[-2, 0, 2, 1], [2, -1, 0, -3]]], dtype=np.int64
    )
    test_updates = np.arange(16, dtype=np.float32).reshape(2, 2, 4) * 1.5

    ref = ReferenceEvaluator(model)
    [result] = ref.run(
        None, {"data": test_data, "indices": test_indices, "updates": test_updates}
    )

    print("Test data: {}".format(test_data.tolist()))
    print("Test indices: {}".format(test_indices.tolist()))
    print("Test updates: {}".format(test_updates.tolist()))
    print("Test output: {}".format(result.tolist()))


if __name__ == "__main__":
    main()
