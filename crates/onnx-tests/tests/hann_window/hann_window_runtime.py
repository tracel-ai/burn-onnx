#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/hann_window/hann_window_runtime.onnx
#
# Generates a Hann window where `size` is a runtime graph input (not a constant).
# This exercises the WindowSize::Runtime codegen path.

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    X = helper.make_tensor_value_info("size", TensorProto.INT64, [])
    # Shape ["N"] is symbolic/dynamic since the size comes from a runtime input.
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N"])

    hann_node = helper.make_node(
        "HannWindow",
        inputs=["size"],
        outputs=["output"],
        name="hann_node",
        periodic=1,
        output_datatype=TensorProto.FLOAT,
    )

    graph = helper.make_graph(
        [hann_node],
        "hann_window_runtime_model",
        [X],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "hann_window_runtime.onnx")
    print("Finished exporting model to hann_window_runtime.onnx")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("hann_window_runtime.onnx")
    result = session.run(None, {"size": np.int64(10)})
    print(f"Test output data shape: {result[0].shape}")
    print(f"Test output data: {result[0].tolist()}")


if __name__ == "__main__":
    main()
