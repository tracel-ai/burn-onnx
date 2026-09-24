#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/range/range_float_mixed.onnx
# Float range with runtime limit and constant start/delta (lifted as static)

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator


def const(name, value):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=numpy_helper.from_array(np.array(value, dtype=np.float32)),
    )


def main():
    range_node = helper.make_node(
        "Range", inputs=["start", "limit", "delta"], outputs=["output"]
    )

    graph_def = helper.make_graph(
        nodes=[const("start", 0.5), const("delta", 0.25), range_node],
        name="RangeFloatMixedGraph",
        inputs=[helper.make_tensor_value_info("limit", TensorProto.FLOAT, [])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
    )

    model_def = helper.make_model(graph_def, producer_name="range_float_mixed")
    model_def.opset_import[0].version = 16

    onnx.save(model_def, "range_float_mixed.onnx")
    print("Model saved to range_float_mixed.onnx")

    sess = ReferenceEvaluator(model_def)
    for limit in [2.0, 1.6]:
        result = sess.run(None, {"limit": np.array(limit, dtype=np.float32)})[0]
        print(f"limit={limit}: {result}")
    result = sess.run(None, {"limit": np.array(2.0, dtype=np.float32)})[0]
    expected = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float32)
    assert np.array_equal(result, expected), f"Mismatch: got {result}"


if __name__ == "__main__":
    main()
