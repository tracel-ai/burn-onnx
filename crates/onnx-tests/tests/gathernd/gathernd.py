#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate ONNX test models for GatherND operator.

Test cases are derived from the ONNX GatherND spec examples.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def compute_gathernd_output_shape(data_shape, indices_shape, batch_dims=0):
    """Compute output shape per the ONNX GatherND spec."""
    q = len(indices_shape)
    b = batch_dims
    k = indices_shape[-1]
    # output_shape = data_shape[:b] + indices_shape[b:q-1] + data_shape[b+k:]
    return list(data_shape[:b]) + list(indices_shape[b:q-1]) + list(data_shape[b+k:])


def make_gathernd_model(name, data_shape, indices, batch_dims=0, dtype=TensorProto.FLOAT):
    """Create an ONNX model for GatherND with constant indices."""
    indices_array = np.array(indices, dtype=np.int64)

    output_shape = compute_gathernd_output_shape(data_shape, indices_array.shape, batch_dims)

    data = helper.make_tensor_value_info("data", dtype, data_shape)
    output = helper.make_tensor_value_info("output", dtype, output_shape)

    indices_init = helper.make_tensor(
        "indices", TensorProto.INT64, indices_array.shape, indices_array.flatten().tolist()
    )

    attrs = {}
    if batch_dims != 0:
        attrs["batch_dims"] = batch_dims

    node = helper.make_node(
        "GatherND",
        inputs=["data", "indices"],
        outputs=["output"],
        **attrs,
    )

    graph = helper.make_graph(
        [node],
        name,
        [data],
        [output],
        initializer=[indices_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    filename = f"{name}.onnx"
    onnx.save(model, filename)
    print(f"Saved {filename}")

    return model, indices_array


def verify_model(model, data, indices):
    """Verify model output using ONNX ReferenceEvaluator."""
    evaluator = ReferenceEvaluator(model)
    result = evaluator.run(None, {"data": data, "indices": indices})
    return result[0]


def main():
    np.random.seed(42)

    # Test 1: Spec Example 1 - batch_dims=0, full index (k==r)
    # data [2,2], indices [2,2] -> output [2]
    model, indices = make_gathernd_model(
        "gathernd",
        data_shape=[2, 2],
        indices=[[0, 0], [1, 1]],
    )
    data = np.array([[0, 1], [2, 3]], dtype=np.float32)
    output = verify_model(model, data, indices)
    print(f"Test 1: data={data.tolist()}, indices={indices.tolist()}")
    print(f"  output={output.tolist()}")
    assert output.tolist() == [0.0, 3.0], f"Expected [0, 3], got {output.tolist()}"

    # Test 2: Spec Example 2 - batch_dims=0, partial index (k<r)
    # data [2,2], indices [2,1] -> output [2,2]
    model, indices = make_gathernd_model(
        "gathernd_partial",
        data_shape=[2, 2],
        indices=[[1], [0]],
    )
    data = np.array([[0, 1], [2, 3]], dtype=np.float32)
    output = verify_model(model, data, indices)
    print(f"Test 2: output={output.tolist()}")
    assert output.tolist() == [[2.0, 3.0], [0.0, 1.0]]

    # Test 3: Spec Example 3 - batch_dims=0, 3D data
    # data [2,2,2], indices [2,2] -> output [2,2]
    model, indices = make_gathernd_model(
        "gathernd_3d",
        data_shape=[2, 2, 2],
        indices=[[0, 1], [1, 0]],
    )
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    output = verify_model(model, data, indices)
    print(f"Test 3: output={output.tolist()}")
    assert output.tolist() == [[2.0, 3.0], [4.0, 5.0]]

    # Test 4: Spec Example 5 - batch_dims=1
    # data [2,2,2], indices [2,1] -> output [2,2]
    model, indices = make_gathernd_model(
        "gathernd_batch1",
        data_shape=[2, 2, 2],
        indices=[[1], [0]],
        batch_dims=1,
    )
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    output = verify_model(model, data, indices)
    print(f"Test 4 (batch_dims=1): output={output.tolist()}")
    assert output.tolist() == [[2.0, 3.0], [4.0, 5.0]]

    # Test 5: Negative indices
    # data [3,2], indices [2,1] with negative index -> output [2,2]
    model, indices = make_gathernd_model(
        "gathernd_neg_idx",
        data_shape=[3, 2],
        indices=[[-1], [0]],
    )
    data = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    output = verify_model(model, data, indices)
    print(f"Test 5 (negative indices): output={output.tolist()}")
    assert output.tolist() == [[4.0, 5.0], [0.0, 1.0]]

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
