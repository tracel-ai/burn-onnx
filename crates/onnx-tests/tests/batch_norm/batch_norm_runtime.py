#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generate a BatchNormalization model where scale, bias, mean, and var
# are graph inputs (runtime tensors), not static initializers.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # BatchNorm with all weight inputs coming from graph inputs (runtime)
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
    scale_info = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3])
    bias_info = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3])
    mean_info = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [3])
    var_info = helper.make_tensor_value_info("var", TensorProto.FLOAT, [3])
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 2, 2]
    )

    graph = helper.make_graph(
        [bn_node],
        "batch_norm_runtime",
        [input_info, scale_info, bias_info, mean_info, var_info],
        [output_info],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)

    # Compute expected output using the ONNX reference implementation
    np.random.seed(42)
    input_data = np.random.randn(1, 3, 2, 2).astype(np.float32)
    scale_data = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    bias_data = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    mean_data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    var_data = np.array([1.0, 2.0, 0.5], dtype=np.float32)

    ref = ReferenceEvaluator(model)
    result = ref.run(
        None,
        {
            "input": input_data,
            "scale": scale_data,
            "bias": bias_data,
            "mean": mean_data,
            "var": var_data,
        },
    )
    output_data = result[0]
    print(f"Input sum: {input_data.sum()}")
    print(f"Output sum: {output_data.sum()}")
    print(f"Output shape: {output_data.shape}")

    onnx.save(model, "batch_norm_runtime.onnx")
    print("Saved batch_norm_runtime.onnx")


if __name__ == "__main__":
    main()
