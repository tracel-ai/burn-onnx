#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/stft/stft_runtime_params.onnx
#
# STFT whose frame_step and frame_length are graph inputs rather than constants, as in
# the ONNX backend test `test_stft`. Run with a non-power-of-two frame_length (12), so
# burn's stft takes its Bluestein path.
#
# Input: [1, 32, 1] real signal, frame_step=4, frame_length=12
# -> n_frames = 1 + (32-12)/4 = 6, onesided n_freqs = 12/2 + 1 = 7
# Output: [1, 6, 7, 2]

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    signal = helper.make_tensor_value_info("signal", TensorProto.FLOAT, [1, 32, 1])
    frame_step = helper.make_tensor_value_info("frame_step", TensorProto.INT64, [])
    frame_length = helper.make_tensor_value_info("frame_length", TensorProto.INT64, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "frames", "freqs", 2])

    node = helper.make_node(
        "STFT",
        inputs=["signal", "frame_step", "", "frame_length"],
        outputs=["output"],
        onesided=1,
    )
    graph = helper.make_graph(
        [node], "stft_runtime_params_model", [signal, frame_step, frame_length], [output]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "stft_runtime_params.onnx")

    np.random.seed(42)
    test_input = np.random.randn(1, 32, 1).astype(np.float32)

    session = ReferenceEvaluator(model)
    (result,) = session.run(
        None,
        {
            "signal": test_input,
            "frame_step": np.array(4, dtype=np.int64),
            "frame_length": np.array(12, dtype=np.int64),
        },
    )
    print(f"Test output shape: {result.shape}")
    print(f"Test output: {result.tolist()}")


if __name__ == "__main__":
    main()
