#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates: lstm_size_one.onnx
# Forward LSTM with initializer weights and input_size=1, hidden_size=1, so the W and
# R slices of one direction both have a size-1 trailing axis. seq_length=3,
# batch_size=2.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 14

INPUT_SIZE = 1
HIDDEN_SIZE = 1
SEQ_LENGTH = 3
BATCH_SIZE = 2
NUM_DIRECTIONS = 1


def ramp(shape, scale, offset):
    """Deterministic values the Rust test reproduces without embedding literals."""
    count = int(np.prod(shape))
    return (np.arange(count, dtype=np.float32) * scale + offset).reshape(shape)


def main():
    np.random.seed(42)

    W = np.random.randn(NUM_DIRECTIONS, 4 * HIDDEN_SIZE, INPUT_SIZE).astype(np.float32)
    R = np.random.randn(NUM_DIRECTIONS, 4 * HIDDEN_SIZE, HIDDEN_SIZE).astype(np.float32)
    B = np.random.randn(NUM_DIRECTIONS, 8 * HIDDEN_SIZE).astype(np.float32) * 0.1

    lstm_node = helper.make_node(
        "LSTM",
        inputs=["input", "W", "R", "B"],
        outputs=["Y", "Y_h"],
        hidden_size=HIDDEN_SIZE,
    )

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE]
    )
    out_Y = helper.make_tensor_value_info(
        "Y",
        TensorProto.FLOAT,
        [SEQ_LENGTH, NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE],
    )
    out_Y_h = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE]
    )

    graph = helper.make_graph(
        [lstm_node],
        "lstm_size_one_graph",
        [inp],
        [out_Y, out_Y_h],
        initializer=[
            numpy_helper.from_array(W, name="W"),
            numpy_helper.from_array(R, name="R"),
            numpy_helper.from_array(B, name="B"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)

    test_input = ramp([SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE], 0.25, -0.5)

    ref = ReferenceEvaluator(model)
    y, y_h = ref.run(None, {"input": test_input})

    print(f"Y shape: {y.shape}, values: {y.ravel().tolist()}")
    print(f"Y_h shape: {y_h.shape}, values: {y_h.ravel().tolist()}")

    onnx.save(model, "lstm_size_one.onnx")
    print("Saved lstm_size_one.onnx")


if __name__ == "__main__":
    main()
