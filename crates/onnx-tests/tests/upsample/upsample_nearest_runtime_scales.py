#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: upsample_nearest_runtime_scales.onnx
#
# Opset 9 with scales as a graph input rather than an initializer, so the
# output size is only known at runtime. This mirrors the upstream ONNX
# backend test test_upsample_nearest.

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    input = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])
    scales = onnx.helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])
    output = onnx.helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 4, 6]
    )

    upsample = onnx.helper.make_node(
        "Upsample",
        inputs=["input", "scales"],
        outputs=["output"],
        name="UpsampleNode",
        mode="nearest",
    )

    graph = onnx.helper.make_graph(
        [upsample], "UpsampleNearestRuntimeScalesModel", [input, scales], [output]
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
    file_name = "upsample_nearest_runtime_scales.onnx"

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    test_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    test_scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    (test_output,) = session.run(None, {"input": test_input, "scales": test_scales})
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output: {repr(test_output)}")
