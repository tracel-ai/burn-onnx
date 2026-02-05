#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates: gru_reverse.onnx
# Reverse GRU with bias, hidden_size=8, input_size=4, seq_length=5, batch_size=2

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 14


def main():
    np.random.seed(43)

    input_size = 4
    hidden_size = 8
    seq_length = 5
    batch_size = 2
    num_directions = 1

    W = np.random.randn(num_directions, 3 * hidden_size, input_size).astype(np.float32) * 0.3
    R = np.random.randn(num_directions, 3 * hidden_size, hidden_size).astype(np.float32) * 0.3
    B = np.random.randn(num_directions, 6 * hidden_size).astype(np.float32) * 0.1

    init_W = numpy_helper.from_array(W, name="W")
    init_R = numpy_helper.from_array(R, name="R")
    init_B = numpy_helper.from_array(B, name="B")

    gru_node = helper.make_node(
        "GRU",
        inputs=["input", "W", "R", "B"],
        outputs=["Y", "Y_h"],
        hidden_size=hidden_size,
        direction="reverse",
        linear_before_reset=1,
    )

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [seq_length, batch_size, input_size]
    )
    out_Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [seq_length, num_directions, batch_size, hidden_size]
    )
    out_Y_h = helper.make_tensor_value_info(
        "Y_h", TensorProto.FLOAT, [num_directions, batch_size, hidden_size]
    )

    graph = helper.make_graph(
        [gru_node],
        "gru_reverse_graph",
        [inp],
        [out_Y, out_Y_h],
        initializer=[init_W, init_R, init_B],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)

    # Validate with ReferenceEvaluator
    np.random.seed(99)
    test_input = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)

    ref = ReferenceEvaluator(model)
    results = ref.run(None, {"input": test_input})

    print(f"Y shape: {results[0].shape}")
    print(f"Y_h shape: {results[1].shape}")
    print(f"Y sum: {results[0].sum():.7f}")
    print(f"Y_h sum: {results[1].sum():.7f}")

    onnx.save(model, "gru_reverse.onnx")
    print("Saved gru_reverse.onnx")


if __name__ == "__main__":
    main()
