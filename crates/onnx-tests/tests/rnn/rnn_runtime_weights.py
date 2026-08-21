#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates: rnn_runtime_weights.onnx
# RNN whose W/R/B arrive as graph inputs rather than initializers, the shape every
# RNN test in the upstream ONNX backend suite uses. hidden_size=3, input_size=2,
# seq_length=2, batch_size=1.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 14

INPUT_SIZE = 2
HIDDEN_SIZE = 3
SEQ_LENGTH = 2
BATCH_SIZE = 1
NUM_DIRECTIONS = 1


def ramp(shape, scale, offset):
    """Deterministic values the Rust test reproduces without embedding literals."""
    count = int(np.prod(shape))
    return (np.arange(count, dtype=np.float32) * scale + offset).reshape(shape)


def main():
    rnn_node = helper.make_node(
        "RNN",
        inputs=["input", "W", "R", "B"],
        outputs=["Y", "Y_h"],
        hidden_size=HIDDEN_SIZE,
    )

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE]
    )
    w = helper.make_tensor_value_info(
        "W", TensorProto.FLOAT, [NUM_DIRECTIONS, 1 * HIDDEN_SIZE, INPUT_SIZE]
    )
    r = helper.make_tensor_value_info(
        "R", TensorProto.FLOAT, [NUM_DIRECTIONS, 1 * HIDDEN_SIZE, HIDDEN_SIZE]
    )
    b = helper.make_tensor_value_info(
        "B", TensorProto.FLOAT, [NUM_DIRECTIONS, 2 * HIDDEN_SIZE]
    )
    out_Y = helper.make_tensor_value_info(
        "Y",
        TensorProto.FLOAT,
        [SEQ_LENGTH, NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE],
    )
    out_Y_h = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE]
    )

    graph = helper.make_graph([rnn_node], "rnn_runtime_weights_graph", [inp, w, r, b], [out_Y, out_Y_h])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)

    test_input = ramp([SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE], 0.25, -0.5)
    test_w = ramp([NUM_DIRECTIONS, 1 * HIDDEN_SIZE, INPUT_SIZE], 0.02, -0.4)
    test_r = ramp([NUM_DIRECTIONS, 1 * HIDDEN_SIZE, HIDDEN_SIZE], 0.015, -0.3)
    test_b = ramp([NUM_DIRECTIONS, 2 * HIDDEN_SIZE], 0.01, -0.1)

    ref = ReferenceEvaluator(model)
    results = ref.run(None, {"input": test_input, "W": test_w, "R": test_r, "B": test_b})

    print(f"Y shape: {results[0].shape}, sum: {results[0].sum():.7f}")
    print(f"Y_h shape: {results[1].shape}, sum: {results[1].sum():.7f}")

    onnx.save(model, "rnn_runtime_weights.onnx")
    print("Saved rnn_runtime_weights.onnx")


if __name__ == "__main__":
    main()
