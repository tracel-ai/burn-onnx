#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/range/range_float_static.onnx
# Float range with fractional constant start and delta (all lifted as static)

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
        nodes=[const("start", 1.5), const("limit", 5.0), const("delta", 0.5), range_node],
        name="RangeFloatStaticGraph",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
    )

    model_def = helper.make_model(graph_def, producer_name="range_float_static")
    model_def.opset_import[0].version = 16

    onnx.save(model_def, "range_float_static.onnx")
    print("Model saved to range_float_static.onnx")

    result = ReferenceEvaluator(model_def).run(None, {})[0]
    print(f"Result: {result}")
    expected = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], dtype=np.float32)
    assert np.array_equal(result, expected), f"Mismatch: got {result}"


if __name__ == "__main__":
    main()
