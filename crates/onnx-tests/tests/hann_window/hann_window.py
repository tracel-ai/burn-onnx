#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/hann_window/hann_window.onnx
#
# Generates a Hann window of size 10 (periodic, float32 output).

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10])

    size_const = helper.make_tensor("size", TensorProto.INT64, [], [10])

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
        "hann_window_model",
        [],
        [Y],
        initializer=[size_const],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "hann_window.onnx")
    print("Finished exporting model to hann_window.onnx")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("hann_window.onnx")
    result = session.run(None, {})
    print(f"Test output data shape: {result[0].shape}")
    print(f"Test output data: {result[0].tolist()}")


if __name__ == "__main__":
    main()
