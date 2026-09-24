#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/range/range_float_runtime.onnx
# Float range with all three bounds as runtime inputs

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    range_node = helper.make_node(
        "Range", inputs=["start", "limit", "delta"], outputs=["output"]
    )

    graph_def = helper.make_graph(
        nodes=[range_node],
        name="RangeFloatRuntimeGraph",
        inputs=[
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [])
            for name in ["start", "limit", "delta"]
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
    )

    model_def = helper.make_model(graph_def, producer_name="range_float_runtime")
    model_def.opset_import[0].version = 16

    onnx.save(model_def, "range_float_runtime.onnx")
    print("Model saved to range_float_runtime.onnx")

    sess = ReferenceEvaluator(model_def)
    # (-1.5, 1.1, 1.3): limit - start is 2.6 in f32, so 2 elements (as in ORT);
    #   subtracting in f64 would give 3
    # (2.0, 0.4, -0.5): negative fractional delta
    # (1.0, 0.0, 0.5): empty
    cases = [
        ((-1.5, 1.1, 1.3), 2),
        ((2.0, 0.4, -0.5), 4),
        ((1.0, 0.0, 0.5), 0),
    ]
    for (start, limit, delta), expected_len in cases:
        feeds = {
            "start": np.array(start, dtype=np.float32),
            "limit": np.array(limit, dtype=np.float32),
            "delta": np.array(delta, dtype=np.float32),
        }
        result = sess.run(None, feeds)[0]
        print(f"start={start}, limit={limit}, delta={delta}: {result!r}")
        assert len(result) == expected_len, f"Mismatch: got {result}"


if __name__ == "__main__":
    main()
