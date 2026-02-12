#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate Split model with num_outputs=3 on dim=10 (not evenly divisible).

Per ONNX spec: split into equal parts, last part smaller if needed.
ceil(10/3) = 4, so splits are [4, 4, 2].
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 18  # num_outputs attribute added in opset 18


def main():
    split_node = helper.make_node(
        "Split",
        ["X"],
        ["Y0", "Y1", "Y2"],
        axis=0,
        num_outputs=3,
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [10, 3])
    Y0 = helper.make_tensor_value_info("Y0", TensorProto.FLOAT, [4, 3])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [4, 3])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph([split_node], "split_uneven", [X], [Y0, Y1, Y2])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])
    onnx.checker.check_model(model)
    onnx.save(model, "split_uneven.onnx")

    # Compute expected outputs
    np.random.seed(42)
    x = np.random.randn(10, 3).astype(np.float32)
    ref = ReferenceEvaluator(model)
    y0, y1, y2 = ref.run(None, {"X": x})
    print("Input X:")
    print(repr(x))
    print(f"\nY0 shape: {y0.shape}")
    print(repr(y0))
    print(f"\nY1 shape: {y1.shape}")
    print(repr(y1))
    print(f"\nY2 shape: {y2.shape}")
    print(repr(y2))


if __name__ == "__main__":
    main()
