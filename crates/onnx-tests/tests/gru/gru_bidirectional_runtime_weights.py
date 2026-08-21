#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates: gru_bidirectional_runtime_weights.onnx and gru_bidirectional_static_weights.onnx
#
# The same bidirectional GRU twice: once with W/R/B as graph inputs, once with the
# identical values as initializers. The Rust test runs both and compares element-wise.
#
# This covers the two-direction branch of the runtime weight loader together with GRU's
# split Wb/Rb bias layout, which no upstream test reaches, and it pins the property that
# split exists for: the build-time snapshot path and the generated runtime path must
# agree. onnx.reference cannot be the oracle here because its GRU raises
# NotImplementedError for num_directions=2, so the static path is the reference.
#
# linear_before_reset=1 keeps Wb and Rb separately observable; under the default 0 they
# are only ever summed, so a swap between them would be invisible.

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

OPSET_VERSION = 14

INPUT_SIZE = 2
HIDDEN_SIZE = 3
SEQ_LENGTH = 2
BATCH_SIZE = 1
NUM_DIRECTIONS = 2


def ramp(shape, scale, offset):
    """Deterministic values the Rust test reproduces without embedding literals."""
    count = int(np.prod(shape))
    return (np.arange(count, dtype=np.float32) * scale + offset).reshape(shape)


def main():
    gru_node = helper.make_node(
        "GRU",
        inputs=["input", "W", "R", "B"],
        outputs=["Y", "Y_h"],
        hidden_size=HIDDEN_SIZE,
        linear_before_reset=1,
        direction="bidirectional",
    )

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE]
    )
    w = helper.make_tensor_value_info(
        "W", TensorProto.FLOAT, [NUM_DIRECTIONS, 3 * HIDDEN_SIZE, INPUT_SIZE]
    )
    r = helper.make_tensor_value_info(
        "R", TensorProto.FLOAT, [NUM_DIRECTIONS, 3 * HIDDEN_SIZE, HIDDEN_SIZE]
    )
    b = helper.make_tensor_value_info(
        "B", TensorProto.FLOAT, [NUM_DIRECTIONS, 6 * HIDDEN_SIZE]
    )
    out_Y = helper.make_tensor_value_info(
        "Y",
        TensorProto.FLOAT,
        [SEQ_LENGTH, NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE],
    )
    out_Y_h = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE]
    )

    runtime = helper.make_model(
        helper.make_graph(
            [gru_node],
            "gru_bidirectional_runtime_weights_graph",
            [inp, w, r, b],
            [out_Y, out_Y_h],
        ),
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(runtime)
    onnx.save(runtime, "gru_bidirectional_runtime_weights.onnx")
    print("Saved gru_bidirectional_runtime_weights.onnx")

    test_w = ramp([NUM_DIRECTIONS, 3 * HIDDEN_SIZE, INPUT_SIZE], 0.02, -0.4)
    test_r = ramp([NUM_DIRECTIONS, 3 * HIDDEN_SIZE, HIDDEN_SIZE], 0.015, -0.3)
    test_b = ramp([NUM_DIRECTIONS, 6 * HIDDEN_SIZE], 0.01, -0.1)

    static = helper.make_model(
        helper.make_graph(
            [gru_node],
            "gru_bidirectional_static_weights_graph",
            [inp],
            [out_Y, out_Y_h],
            initializer=[
                numpy_helper.from_array(test_w, name="W"),
                numpy_helper.from_array(test_r, name="R"),
                numpy_helper.from_array(test_b, name="B"),
            ],
        ),
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(static)
    onnx.save(static, "gru_bidirectional_static_weights.onnx")
    print("Saved gru_bidirectional_static_weights.onnx")


if __name__ == "__main__":
    main()
