#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: softsign.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    # Build graph: Y = Softsign(X)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])

    softsign_node = helper.make_node("Softsign", inputs=["X"], outputs=["Y"])

    graph = helper.make_graph([softsign_node], "softsign_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    file_name = "softsign.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    # Compute expected outputs using the reference evaluator
    np.random.seed(42)
    test_input = np.random.randn(2, 3).astype(np.float32)

    ref = ReferenceEvaluator(model)
    [output] = ref.run(None, {"X": test_input})

    print("Test input data: {}".format(test_input))
    print("Test input data shape: {}".format(test_input.shape))
    print("Test output data shape: {}".format(output.shape))
    print("Test output: {}".format(output))


if __name__ == "__main__":
    main()
