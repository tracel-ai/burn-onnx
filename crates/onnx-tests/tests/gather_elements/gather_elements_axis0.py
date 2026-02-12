#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate GatherElements model with axis=0 on [3, 3] data, [2, 3] indices."""

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    node = helper.make_node(
        "GatherElements",
        ["data", "indices"],
        ["output"],
        axis=0,
    )

    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph([node], "gather_elements_axis0", [data, indices], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])
    onnx.checker.check_model(model)
    onnx.save(model, "gather_elements_axis0.onnx")

    # Compute expected outputs
    np.random.seed(42)
    data_val = np.random.randn(3, 3).astype(np.float32)
    indices_val = np.array([[0, 2, 1], [2, 0, 0]], dtype=np.int64)
    ref = ReferenceEvaluator(model)
    (output_val,) = ref.run(None, {"data": data_val, "indices": indices_val})
    print("Input data:")
    print(repr(data_val))
    print("\nInput indices:")
    print(repr(indices_val))
    print("\nExpected output:")
    print(repr(output_val))


if __name__ == "__main__":
    main()
