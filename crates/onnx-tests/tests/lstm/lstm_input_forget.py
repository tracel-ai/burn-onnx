#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# Generates: lstm_input_forget.onnx
# LSTM with input_forget=1 (f_t = 1 - i_t) and W/R/B as initializers, so the
# forget gate's slice of the packed weights must be ignored. hidden_size=3,
# input_size=2, seq_length=3, batch_size=1.
#
# onnx.reference ignores input_forget, so the expected values come from a numpy
# implementation of the coupled cell, checked against onnxruntime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

OPSET_VERSION = 14

INPUT_SIZE = 2
HIDDEN_SIZE = 3
SEQ_LENGTH = 3
BATCH_SIZE = 1


def ramp(shape, scale, offset):
    """Deterministic values the Rust test reproduces without embedding literals."""
    count = int(np.prod(shape))
    return (np.arange(count, dtype=np.float32) * scale + offset).reshape(shape)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def reference_lstm(x, w, r, b, input_forget):
    """Forward LSTM with ONNX's [i, o, f, c] gate packing."""
    h = np.zeros([BATCH_SIZE, HIDDEN_SIZE], dtype=np.float32)
    c = np.zeros([BATCH_SIZE, HIDDEN_SIZE], dtype=np.float32)
    bias = b[0, : 4 * HIDDEN_SIZE] + b[0, 4 * HIDDEN_SIZE :]
    ys = []
    for t in range(SEQ_LENGTH):
        gates = x[t] @ w[0].T + h @ r[0].T + bias
        i, o, f, g = np.split(gates, 4, axis=-1)
        i = sigmoid(i)
        f = 1.0 - i if input_forget else sigmoid(f)
        c = f * c + i * np.tanh(g)
        h = sigmoid(o) * np.tanh(c)
        ys.append(h)
    return np.stack(ys)[:, None], h[None], c[None]


def main():
    w = ramp([1, 4 * HIDDEN_SIZE, INPUT_SIZE], 0.03, -0.35)
    r = ramp([1, 4 * HIDDEN_SIZE, HIDDEN_SIZE], -0.02, 0.4)
    b = ramp([1, 8 * HIDDEN_SIZE], 0.015, -0.2)

    lstm_node = helper.make_node(
        "LSTM",
        inputs=["input", "W", "R", "B"],
        outputs=["Y", "Y_h", "Y_c"],
        hidden_size=HIDDEN_SIZE,
        input_forget=1,
    )

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE]
    )
    out_y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [SEQ_LENGTH, 1, BATCH_SIZE, HIDDEN_SIZE]
    )
    out_y_h = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [1, BATCH_SIZE, HIDDEN_SIZE]
    )
    out_y_c = helper.make_tensor_value_info(
        "Y_c", TensorProto.FLOAT, [1, BATCH_SIZE, HIDDEN_SIZE]
    )

    graph = helper.make_graph(
        [lstm_node],
        "lstm_input_forget_graph",
        [inp],
        [out_y, out_y_h, out_y_c],
        initializer=[
            numpy_helper.from_array(w, name="W"),
            numpy_helper.from_array(r, name="R"),
            numpy_helper.from_array(b, name="B"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)

    test_input = ramp([SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE], 0.3, -0.6)

    expected = reference_lstm(test_input, w, r, b, input_forget=True)
    uncoupled = reference_lstm(test_input, w, r, b, input_forget=False)
    assert not np.allclose(expected[2], uncoupled[2], atol=1e-3), (
        "coupling must change the result, or the test cannot tell the two apart"
    )

    session = ort.InferenceSession(model.SerializeToString())
    actual = session.run(None, {"input": test_input})
    for name, want, got in zip(["Y", "Y_h", "Y_c"], expected, actual):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=name)

    np.set_printoptions(precision=7)
    for name, value in zip(["Y", "Y_h", "Y_c"], expected):
        print(f"{name} shape: {value.shape}, values: {value.flatten().tolist()}")

    onnx.save(model, "lstm_input_forget.onnx")
    print("Saved lstm_input_forget.onnx")


if __name__ == "__main__":
    main()
