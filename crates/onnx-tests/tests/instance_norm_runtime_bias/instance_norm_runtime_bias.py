#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: instance_norm_runtime_bias.onnx
#
# Constant scale (Constant node) with a runtime bias (graph input). Scale and
# bias must be lifted together, so this takes the runtime path rather than
# lifting only the scale.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    np.random.seed(42)

    scale_node = helper.make_node(
        "Constant",
        [],
        ["scale"],
        value=numpy_helper.from_array(np.array([0.5, 2.0], dtype=np.float32)),
    )
    node = helper.make_node(
        "InstanceNormalization",
        inputs=["input", "scale", "bias"],
        outputs=["output"],
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [scale_node, node],
        "instance_norm_runtime_bias",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [2]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)

    file_name = "instance_norm_runtime_bias.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")

    test_input = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    test_bias = np.array([1.0, -1.0], dtype=np.float32)
    (output,) = ReferenceEvaluator(model).run(
        None, {"input": test_input, "bias": test_bias}
    )
    print(f"Test output: {repr(output)}")


if __name__ == "__main__":
    main()
