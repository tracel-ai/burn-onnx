#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/stft/stft_large_non_pow2.onnx
#
# Large non-power-of-two STFT (n_fft=400, Whisper's size) through burn-onnx's
# matrix-DFT codegen path. The input is a pure cosine at an integer bin, so the
# exact output is known: N/2 at that bin and 0 everywhere else. That pins the
# accuracy of near-zero bins, where an f32 matmul accumulates O(N * eps) error.
#
# Input: [1, 1200, 1] real signal, x[n] = cos(2 pi * 7 * n / 400)
# frame_step=400, frame_length=400 -> n_frames = 1 + (1200-400)/400 = 3
# onesided=1 -> n_freqs = 400/2 + 1 = 201
# Output: [1, 3, 201, 2]

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

N_FFT = 400
BIN = 7
LENGTH = 1200


def main():
    X = helper.make_tensor_value_info("signal", TensorProto.FLOAT, [1, LENGTH, 1])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 201, 2])

    frame_step = numpy_helper.from_array(
        np.array(N_FFT, dtype=np.int64), name="frame_step"
    )
    frame_length = numpy_helper.from_array(
        np.array(N_FFT, dtype=np.int64), name="frame_length"
    )

    stft_node = helper.make_node(
        "STFT",
        inputs=["signal", "frame_step", "", "frame_length"],
        outputs=["output"],
        name="stft_node",
        onesided=1,
    )

    graph = helper.make_graph(
        [stft_node],
        "stft_large_non_pow2_model",
        [X],
        [Y],
        initializer=[frame_step, frame_length],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "stft_large_non_pow2.onnx")
    print("Finished exporting model to stft_large_non_pow2.onnx")

    n = np.arange(LENGTH, dtype=np.float64)
    test_input = np.cos(2 * np.pi * BIN * n / N_FFT).astype(np.float32)
    test_input = test_input.reshape(1, LENGTH, 1)

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("stft_large_non_pow2.onnx")
    result = session.run(None, {"signal": test_input})[0]

    expected = np.zeros((1, 3, 201, 2), dtype=np.float64)
    expected[0, :, BIN, 0] = N_FFT / 2
    max_err = np.abs(result - expected).max()
    print(f"Test output shape: {result.shape}")
    print(f"Max abs error vs analytic DFT: {max_err:.3e}")
    assert max_err < 1e-3, max_err


if __name__ == "__main__":
    main()
