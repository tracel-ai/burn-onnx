#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate GatherElements model with axis=1 on [2, 3, 4] data, [2, 2, 4] indices."""

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
        axis=1,
    )

    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2, 4])

    graph = helper.make_graph([node], "gather_elements_3d", [data, indices], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])
    onnx.checker.check_model(model)
    onnx.save(model, "gather_elements_3d.onnx")

    # Compute expected outputs
    np.random.seed(42)
    data_val = np.random.randn(2, 3, 4).astype(np.float32)
    indices_val = np.random.randint(0, 3, size=(2, 2, 4)).astype(np.int64)
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
