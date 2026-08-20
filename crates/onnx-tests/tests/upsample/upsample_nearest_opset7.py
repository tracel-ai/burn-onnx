#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
#   "onnxruntime==1.20.1",
# ]
# ///

# used to generate model: upsample_nearest_opset7.onnx
#
# Opset 7 carries the scales as a float-list attribute, one entry per input
# dimension. This is the form older exports (e.g. fastdepth) use.
#
# Note: ONNX ReferenceEvaluator implements only the opset 9 form of Upsample (scales as an
# input), and its implementation is np.repeat, which raises on non-integer scales. onnxruntime
# is therefore the only usable oracle for opset 7, and for any future non-integer scale test.

import numpy as np
import onnx
import onnx.helper
import onnxruntime as ort
from onnx import TensorProto


def build_model():
    input = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 3])
    output = onnx.helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2, 4, 6]
    )

    upsample = onnx.helper.make_node(
        "Upsample",
        inputs=["input"],
        outputs=["output"],
        name="UpsampleNode",
        mode="nearest",
        scales=[1.0, 1.0, 2.0, 2.0],
    )

    graph = onnx.helper.make_graph(
        [upsample], "UpsampleNearestOpset7Model", [input], [output]
    )

    return onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 7)],
        graph=graph,
        producer_name="ONNX_Generator",
        ir_version=8,
    )


if __name__ == "__main__":
    np.random.seed(42)
    np.set_printoptions(precision=8)

    onnx_model = build_model()
    file_name = "upsample_nearest_opset7.onnx"

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    test_input = np.arange(1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)
    print(f"Test input data shape: {test_input.shape}")
    session = ort.InferenceSession(file_name)
    (test_output,) = session.run(None, {"input": test_input})
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output: {repr(test_output)}")
