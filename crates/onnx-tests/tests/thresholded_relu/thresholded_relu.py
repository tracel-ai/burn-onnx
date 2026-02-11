#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: thresholded_relu.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    # Build graph: Y = ThresholdedRelu(X, alpha=2.0)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])

    node = helper.make_node(
        "ThresholdedRelu", inputs=["X"], outputs=["Y"], alpha=2.0
    )

    graph = helper.make_graph([node], "thresholded_relu_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    file_name = "thresholded_relu.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    # Compute expected outputs using the reference evaluator
    # Hand-crafted input: includes boundary value (2.0 == alpha), above (2.5), and below (-1.0, 0.0, 1.5)
    test_input = np.array([[-1.0, 0.0, 2.0], [2.5, 1.5, 5.0]], dtype=np.float32)

    ref = ReferenceEvaluator(model)
    [output] = ref.run(None, {"X": test_input})

    print("Test input data: {}".format(test_input))
    print("Test input data shape: {}".format(test_input.shape))
    print("Test output data shape: {}".format(output.shape))
    print("Test output: {}".format(output))


if __name__ == "__main__":
    main()
