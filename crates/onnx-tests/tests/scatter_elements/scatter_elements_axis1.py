#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: scatter_elements_axis1.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [3, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 3])

    node = helper.make_node(
        "ScatterElements",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        axis=1,
    )

    graph = helper.make_graph(
        [node], "scatter_elements_axis1_graph", [data, indices, updates], [output]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_elements_axis1.onnx")

    test_data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    test_indices = np.array([[2, 0], [1, 2], [0, 1]], dtype=np.int64)
    test_updates = np.array(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32
    )

    ref = ReferenceEvaluator(model)
    [result] = ref.run(
        None, {"data": test_data, "indices": test_indices, "updates": test_updates}
    )

    print("Test data: {}".format(test_data))
    print("Test indices: {}".format(test_indices))
    print("Test updates: {}".format(test_updates))
    print("Test output: {}".format(result))


if __name__ == "__main__":
    main()
