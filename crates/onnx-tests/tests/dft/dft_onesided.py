#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/dft/dft_onesided.onnx
#
# Forward real DFT with onesided=1.
# Input shape: [1, 8, 1] (batch=1, signal_length=8, real=1)
# Output shape: [1, 5, 2] (batch=1, N/2+1=5, complex=2)

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 1])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 2])

    dft_node = helper.make_node(
        "DFT",
        inputs=["input"],
        outputs=["output"],
        name="dft_node",
        inverse=0,
        onesided=1,
    )

    graph = helper.make_graph(
        [dft_node],
        "dft_onesided_model",
        [X],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10

    onnx.checker.check_model(model)
    onnx.save(model, "dft_onesided.onnx")
    print("Finished exporting model to dft_onesided.onnx")

    # Test with a simple signal: [1, 2, 3, 4, 5, 6, 7, 8]
    test_input = np.array([[[1], [2], [3], [4], [5], [6], [7], [8]]], dtype=np.float32)
    print(f"Test input data shape: {test_input.shape}")
    print(f"Test input data: {test_input.tolist()}")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("dft_onesided.onnx")
    result = session.run(None, {"input": test_input})
    print(f"Test output data shape: {result[0].shape}")
    print(f"Test output data: {result[0].tolist()}")


if __name__ == "__main__":
    main()
