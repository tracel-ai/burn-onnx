#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with a symbolic batch dimension and concrete spatial dims.

This is the standard pattern: input shape [N, 3, 224, 224] where N is symbolic.
Reproduces issue #62: symbolic dimensions cause static_shape to be None,
even though 3 of 4 dims are concrete.
"""

import onnx
from onnx import helper, TensorProto


def create_symbolic_batch_model():
    # Shape: [N, 3, 224, 224] - symbolic batch, concrete spatial dims
    input1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, ["N", 3, 224, 224]
    )

    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["N", 3, 224, 224]
    )

    nodes = [
        helper.make_node("Relu", ["input1"], ["output"], name="relu"),
    ]

    graph = helper.make_graph(
        nodes,
        "symbolic_batch_model",
        [input1],
        [output],
    )

    model = helper.make_model(
        graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)]
    )

    onnx.checker.check_model(model)
    return model


def main():
    model = create_symbolic_batch_model()
    output_path = "../fixtures/symbolic_batch_dim.onnx"
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")
    print(f"Input shape: [N, 3, 224, 224] (symbolic batch dim)")


if __name__ == "__main__":
    main()
