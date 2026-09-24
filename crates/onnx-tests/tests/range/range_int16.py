#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/range/range_int16.onnx
# Int16 range with negative start, all constants (lifted as static)

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator


def const(name, value):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=numpy_helper.from_array(np.array(value, dtype=np.int16)),
    )


def main():
    range_node = helper.make_node(
        "Range", inputs=["start", "limit", "delta"], outputs=["output"]
    )

    graph_def = helper.make_graph(
        nodes=[const("start", -3), const("limit", 3), const("delta", 2), range_node],
        name="RangeInt16Graph",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT16, [None])],
    )

    model_def = helper.make_model(graph_def, producer_name="range_int16")
    model_def.opset_import[0].version = 16

    onnx.save(model_def, "range_int16.onnx")
    print("Model saved to range_int16.onnx")

    result = ReferenceEvaluator(model_def).run(None, {})[0]
    print(f"Result: {result!r}")
    expected = np.array([-3, -1, 1], dtype=np.int16)
    assert np.array_equal(result, expected), f"Mismatch: got {result}"


if __name__ == "__main__":
    main()
