#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: scatter_elements_int.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    # Integer data, which is otherwise only snapshot-tested.
    data = helper.make_tensor_value_info("data", TensorProto.INT64, [3, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3])
    updates = helper.make_tensor_value_info("updates", TensorProto.INT64, [2, 3])
    output = helper.make_tensor_value_info("output", TensorProto.INT64, [3, 3])

    node = helper.make_node(
        "ScatterElements",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        axis=0,
    )

    graph = helper.make_graph(
        [node], "scatter_elements_int_graph", [data, indices, updates], [output]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_elements_int.onnx")

    test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    test_indices = np.array([[1, 0, 2], [0, 2, -2]], dtype=np.int64)
    test_updates = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)

    ref = ReferenceEvaluator(model)
    [result] = ref.run(
        None, {"data": test_data, "indices": test_indices, "updates": test_updates}
    )

    print("Test output: {}".format(result.tolist()))


if __name__ == "__main__":
    main()
