#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates a Pad model that pads a 4D tensor on ALL dimensions
# (including batch and channel), testing N-dimensional padding support.

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator
from pathlib import Path


def main():
    inputs = [make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 3, 3])]
    outputs = [make_tensor_value_info("output", TensorProto.FLOAT, [None, None, None, None])]

    # Pad format: [begin_d0, begin_d1, begin_d2, begin_d3, end_d0, end_d1, end_d2, end_d3]
    # Pad batch by (1, 0), channel by (0, 1), height by (1, 1), width by (2, 2)
    pads = numpy_helper.from_array(
        np.array([1, 0, 1, 2, 0, 1, 1, 2]).astype(np.int64), name="pads"
    )
    constant_value = numpy_helper.from_array(
        np.array([0.0]).astype(np.float32), name="constant_value"
    )

    node = make_node(
        "Pad",
        inputs=["input", "pads", "constant_value"],
        outputs=["output"],
        mode="constant",
    )

    graph = make_graph(
        nodes=[node],
        name="PadNdimGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=[pads, constant_value],
    )

    model = make_model(graph)
    check_model(model)

    # Verify with reference evaluator
    input_data = np.arange(1, 19, dtype=np.float32).reshape(1, 2, 3, 3)
    sess = ReferenceEvaluator(model)
    result = sess.run(None, {"input": input_data})[0]

    # Input shape: [1, 2, 3, 3]
    # After padding: [1+1+0, 2+0+1, 3+1+1, 3+2+2] = [2, 3, 5, 7]
    assert result.shape == (2, 3, 5, 7), f"Expected shape (2, 3, 5, 7), got {result.shape}"
    print(f"Output shape: {result.shape}")
    print(f"Output:\n{result}")

    onnx.save(model, Path(__file__).with_name("pad_ndim.onnx"))
    print("Saved pad_ndim.onnx")


if __name__ == "__main__":
    main()
