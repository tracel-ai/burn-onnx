#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/range/range_double_mixed.onnx
# Double range with runtime limit and constant start/delta (lifted as static)

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator


def const(name, value):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=numpy_helper.from_array(np.array(value, dtype=np.float64)),
    )


def main():
    range_node = helper.make_node(
        "Range", inputs=["start", "limit", "delta"], outputs=["output"]
    )

    graph_def = helper.make_graph(
        nodes=[const("start", -0.5), const("delta", 0.125), range_node],
        name="RangeDoubleMixedGraph",
        inputs=[helper.make_tensor_value_info("limit", TensorProto.DOUBLE, [])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.DOUBLE, [None])],
    )

    model_def = helper.make_model(graph_def, producer_name="range_double_mixed")
    model_def.opset_import[0].version = 16

    onnx.save(model_def, "range_double_mixed.onnx")
    print("Model saved to range_double_mixed.onnx")

    result = ReferenceEvaluator(model_def).run(
        None, {"limit": np.array(0.0, dtype=np.float64)}
    )[0]
    print(f"Result: {result!r}")
    expected = np.array([-0.5, -0.375, -0.25, -0.125], dtype=np.float64)
    assert np.array_equal(result, expected), f"Mismatch: got {result}"


if __name__ == "__main__":
    main()
