#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/expand/expand_shorter_shape.onnx
#
# Tests ONNX Expand with a shape shorter than the input rank. The shape is
# right-aligned against the input, so the output keeps the input's rank:
# Expand(x: [2, 1], shape: [3]) -> [2, 3].
# Covers both a constant shape and a runtime shape input.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main() -> None:
    shape_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["static_shape"],
        value=helper.make_tensor("static_shape_val", TensorProto.INT64, [1], [3]),
    )
    static_expand = helper.make_node(
        "Expand", inputs=["x", "static_shape"], outputs=["static_out"]
    )
    runtime_expand = helper.make_node(
        "Expand", inputs=["x", "shape"], outputs=["runtime_out"]
    )

    graph_def = helper.make_graph(
        nodes=[shape_node, static_expand, runtime_expand],
        name="ExpandShorterShapeGraph",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 1]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info("static_out", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("runtime_out", TensorProto.FLOAT, [2, 3]),
        ],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="expand_shorter_shape",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model_def)

    onnx_name = "expand_shorter_shape.onnx"
    onnx.save(model_def, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    x = np.array([[1.0], [2.0]], dtype=np.float32)
    shape = np.array([3], dtype=np.int64)
    session = ReferenceEvaluator(onnx_name, verbose=0)
    static_out, runtime_out = session.run(None, {"x": x, "shape": shape})
    print(f"Static output:\n{static_out}")
    print(f"Runtime output:\n{runtime_out}")

    expected = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    assert np.array_equal(static_out, expected)
    assert np.array_equal(runtime_out, expected)


if __name__ == "__main__":
    main()
