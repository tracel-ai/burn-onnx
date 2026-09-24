#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/range/range_int32_mixed.onnx
# Int32 range with runtime limit and constant start/delta (lifted as static)

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator


def const(name, value):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=numpy_helper.from_array(np.array(value, dtype=np.int32)),
    )


def main():
    range_node = helper.make_node(
        "Range", inputs=["start", "limit", "delta"], outputs=["output"]
    )

    graph_def = helper.make_graph(
        nodes=[const("start", 1), const("delta", 3), range_node],
        name="RangeInt32MixedGraph",
        inputs=[helper.make_tensor_value_info("limit", TensorProto.INT32, [])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT32, [None])],
    )

    model_def = helper.make_model(graph_def, producer_name="range_int32_mixed")
    model_def.opset_import[0].version = 16

    onnx.save(model_def, "range_int32_mixed.onnx")
    print("Model saved to range_int32_mixed.onnx")

    result = ReferenceEvaluator(model_def).run(
        None, {"limit": np.array(11, dtype=np.int32)}
    )[0]
    print(f"Result: {result}")
    expected = np.array([1, 4, 7, 10], dtype=np.int32)
    assert np.array_equal(result, expected), f"Mismatch: got {result}"


if __name__ == "__main__":
    main()
