#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generate a BatchNormalization model where scale and bias are static
# initializers but mean and var are graph inputs (runtime tensors).
# This tests the "partial constant" case — the Runtime path should be
# used because not all weight inputs are static.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # scale and bias are initializers (static)
    init_scale = numpy_helper.from_array(
        np.array([1.0, 2.0, 0.5], dtype=np.float32), name="scale"
    )
    init_bias = numpy_helper.from_array(
        np.array([0.0, 1.0, -1.0], dtype=np.float32), name="bias"
    )

    bn_node = helper.make_node(
        "BatchNormalization",
        ["input", "scale", "bias", "mean", "var"],
        ["output"],
        epsilon=1e-5,
        momentum=0.9,
    )

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 2, 2]
    )
    # mean and var are graph inputs (runtime)
    mean_info = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [3])
    var_info = helper.make_tensor_value_info("var", TensorProto.FLOAT, [3])
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 2, 2]
    )

    graph = helper.make_graph(
        [bn_node],
        "batch_norm_partial_constant",
        [input_info, mean_info, var_info],
        [output_info],
        initializer=[init_scale, init_bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)

    # Compute expected output
    np.random.seed(42)
    input_data = np.random.randn(1, 3, 2, 2).astype(np.float32)
    mean_data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    var_data = np.array([1.0, 2.0, 0.5], dtype=np.float32)

    ref = ReferenceEvaluator(model)
    result = ref.run(
        None,
        {
            "input": input_data,
            "mean": mean_data,
            "var": var_data,
        },
    )
    output_data = result[0]
    print(f"Input sum: {input_data.sum()}")
    print(f"Output sum: {output_data.sum()}")
    print(f"Output shape: {output_data.shape}")

    onnx.save(model, "batch_norm_partial_constant.onnx")
    print("Saved batch_norm_partial_constant.onnx")


if __name__ == "__main__":
    main()
