#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Shape(input) -> Slice to extract dims [1:3]
    # Reproduces: Shape -> Slice with constant starts/ends/axes/steps

    shape_node = helper.make_node("Shape", ["input"], ["shape_out"])

    # Slice constants: starts=1, ends=3, axes=0, steps=1
    const_axes = helper.make_node(
        "Constant",
        [],
        ["axes"],
        value=numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes"),
    )
    const_starts = helper.make_node(
        "Constant",
        [],
        ["starts"],
        value=numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts"),
    )
    const_ends = helper.make_node(
        "Constant",
        [],
        ["ends"],
        value=numpy_helper.from_array(np.array([3], dtype=np.int64), name="ends"),
    )
    const_steps = helper.make_node(
        "Constant",
        [],
        ["steps"],
        value=numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps"),
    )

    slice_node = helper.make_node(
        "Slice", ["shape_out", "starts", "ends", "axes", "steps"], ["output"]
    )

    # input has dynamic first dim
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2, 3, 1])
    out = helper.make_tensor_value_info("output", TensorProto.INT64, [None])

    graph = helper.make_graph(
        [shape_node, const_axes, const_starts, const_ends, const_steps, slice_node],
        "slice_shape_test",
        [inp],
        [out],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx_name = "slice_shape.onnx"
    onnx.save(model, onnx_name)
    print(f"Successfully exported model to {onnx_name}")

    # Test
    sample_input = np.random.randn(1, 2, 3, 1).astype(np.float32)
    session = ReferenceEvaluator(onnx_name)
    outputs = session.run(None, {"input": sample_input})
    print(f"Sample input tensor shape: {sample_input.shape}")
    print(f"Model output tensor: {outputs[0]}")


if __name__ == "__main__":
    main()
