#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: group_norm_runtime_bias.onnx
#
# Constant scale (initializer) with a runtime bias (graph input). Scale and bias
# must be lifted together, so this takes the runtime path rather than lifting
# only the scale.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    np.random.seed(42)

    scale = numpy_helper.from_array(
        np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32), name="scale"
    )
    node = helper.make_node(
        "GroupNormalization",
        inputs=["input", "scale", "bias"],
        outputs=["output"],
        num_groups=2,
    )
    graph = helper.make_graph(
        [node],
        "group_norm_runtime_bias",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 2, 2]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 2, 2])],
        initializer=[scale],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    onnx.checker.check_model(model)

    file_name = "group_norm_runtime_bias.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")

    test_input = np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2)
    test_bias = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    (output,) = ReferenceEvaluator(model).run(
        None, {"input": test_input, "bias": test_bias}
    )
    print(f"Test output: {repr(output)}")


if __name__ == "__main__":
    main()
