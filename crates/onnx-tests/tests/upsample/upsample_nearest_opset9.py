#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: upsample_nearest_opset9.onnx
#
# Opset 9 moved scales from an attribute to an input. Here it is an
# initializer, so the scales are still known at codegen time.

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator


def build_model():
    input = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 3])
    output = onnx.helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2, 6, 6]
    )

    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 3.0, 2.0], dtype=np.float32), name="scales"
    )

    upsample = onnx.helper.make_node(
        "Upsample",
        inputs=["input", "scales"],
        outputs=["output"],
        name="UpsampleNode",
        mode="nearest",
    )

    graph = onnx.helper.make_graph(
        [upsample],
        "UpsampleNearestOpset9Model",
        [input],
        [output],
        initializer=[scales],
    )

    return onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 9)],
        graph=graph,
        producer_name="ONNX_Generator",
        ir_version=8,
    )


if __name__ == "__main__":
    np.random.seed(42)
    np.set_printoptions(precision=8)

    onnx_model = build_model()
    file_name = "upsample_nearest_opset9.onnx"

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    test_input = np.arange(1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    (test_output,) = session.run(None, {"input": test_input})
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output: {repr(test_output)}")
