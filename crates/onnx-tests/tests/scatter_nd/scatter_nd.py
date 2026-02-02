#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate ONNX models for ScatterND operator tests."""

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def make_scatter_nd_model(name, data_shape, indices, updates_shape, reduction="none"):
    """Create an ONNX model for ScatterND."""
    indices_array = np.array(indices, dtype=np.int64)

    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)

    indices_init = helper.make_tensor(
        "indices", TensorProto.INT64, indices_array.shape, indices_array.flatten().tolist()
    )

    attrs = {}
    if reduction != "none":
        attrs["reduction"] = reduction

    node = helper.make_node(
        "ScatterND",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        **attrs,
    )

    graph = helper.make_graph(
        [node],
        name,
        [data, updates],
        [output],
        initializer=[indices_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    filename = f"{name}.onnx"
    onnx.save(model, filename)
    print(f"Saved {filename}")

    return model, indices_array


def verify_model(model, data, indices, updates):
    """Verify model output using ONNX ReferenceEvaluator."""
    evaluator = ReferenceEvaluator(model)
    result = evaluator.run(None, {"data": data, "indices": indices, "updates": updates})
    return result[0]


def main():
    np.random.seed(42)

    # Test 1: 1D simple scatter (Example 1 from spec)
    model, indices = make_scatter_nd_model(
        "scatter_nd",
        data_shape=[8],
        indices=[[4], [3], [1], [7]],
        updates_shape=[4],
    )
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    updates = np.array([9, 10, 11, 12], dtype=np.float32)
    output = verify_model(model, data, indices, updates)
    print(f"Test 1 - input: {data}")
    print(f"Test 1 - updates: {updates}")
    print(f"Test 1 - output: {output}")
    # Expected: [1, 11, 3, 10, 9, 6, 7, 12]

    # Test 2: 2D slice update (indices select rows)
    model2, indices2 = make_scatter_nd_model(
        "scatter_nd_2d",
        data_shape=[4, 4],
        indices=[[0], [2]],
        updates_shape=[2, 4],
    )
    data2 = np.ones((4, 4), dtype=np.float32)
    updates2 = np.array([[5, 5, 5, 5], [6, 6, 6, 6]], dtype=np.float32)
    output2 = verify_model(model2, data2, indices2, updates2)
    print(f"Test 2 - output:\n{output2}")
    # Expected: [[5,5,5,5],[1,1,1,1],[6,6,6,6],[1,1,1,1]]

    # Test 3: with add reduction
    model3, indices3 = make_scatter_nd_model(
        "scatter_nd_add",
        data_shape=[8],
        indices=[[4], [3], [1], [7]],
        updates_shape=[4],
        reduction="add",
    )
    data3 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    updates3 = np.array([9, 10, 11, 12], dtype=np.float32)
    output3 = verify_model(model3, data3, indices3, updates3)
    print(f"Test 3 (add) - output: {output3}")
    # Expected: [1, 13, 3, 14, 14, 6, 7, 20]

    # Test 4: with mul reduction
    model4, indices4 = make_scatter_nd_model(
        "scatter_nd_mul",
        data_shape=[8],
        indices=[[4], [3], [1], [7]],
        updates_shape=[4],
        reduction="mul",
    )
    data4 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    updates4 = np.array([9, 10, 11, 12], dtype=np.float32)
    output4 = verify_model(model4, data4, indices4, updates4)
    print(f"Test 4 (mul) - output: {output4}")
    # Expected: [1, 22, 3, 40, 45, 6, 7, 96]

    # Test 5: with max reduction
    # Note: ReferenceEvaluator has a bug with max/min on 1D, so we verify manually
    model5, indices5 = make_scatter_nd_model(
        "scatter_nd_max",
        data_shape=[8],
        indices=[[4], [3], [1], [7]],
        updates_shape=[4],
        reduction="max",
    )
    # Expected: [1, max(2,11)=11, 3, max(4,10)=10, max(5,9)=9, 6, 7, max(8,12)=12]
    print("Test 5 (max) - expected: [1, 11, 3, 10, 9, 6, 7, 12]")

    # Test 6: with min reduction
    model6, indices6 = make_scatter_nd_model(
        "scatter_nd_min",
        data_shape=[8],
        indices=[[4], [3], [1], [7]],
        updates_shape=[4],
        reduction="min",
    )
    # Expected: [1, min(2,11)=2, 3, min(4,10)=4, min(5,9)=5, 6, 7, min(8,12)=8]
    print("Test 6 (min) - expected: [1, 2, 3, 4, 5, 6, 7, 8]")


if __name__ == "__main__":
    main()
