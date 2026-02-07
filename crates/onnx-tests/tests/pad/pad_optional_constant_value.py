#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates a Pad model where constant_value (input[2]) is an empty string "",
# simulating the real-world case of an optional input not provided.
# The expected behavior is to use the default constant_value of 0.0.

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator


def main() -> None:
    # Input: 2x3 float32 tensor
    input_info = make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output_info = make_tensor_value_info("output", TensorProto.FLOAT, [4, 5])

    # Pads as initializer: [1, 1, 1, 1] means pad 1 on each side for both dims
    pads_init = numpy_helper.from_array(
        np.array([1, 1, 1, 1], dtype=np.int64), name="pads"
    )

    # Pad node with 3 inputs: data, pads, "" (empty = optional constant_value)
    pad_node = make_node(
        "Pad",
        inputs=["input", "pads", ""],
        outputs=["output"],
        mode="constant",
    )

    graph = make_graph(
        nodes=[pad_node],
        name="PadOptionalConstantValueGraph",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[pads_init],
    )

    model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    check_model(model)

    # Verify with reference evaluator
    sess = ReferenceEvaluator(model)
    test_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = sess.run(None, {"input": test_input})[0]
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 0.0],
            [0.0, 4.0, 5.0, 6.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result, expected)
    print("Reference evaluator test passed")

    onnx.save(model, Path(__file__).with_name("pad_optional_constant_value.onnx"))
    print("Model saved")


if __name__ == "__main__":
    main()
