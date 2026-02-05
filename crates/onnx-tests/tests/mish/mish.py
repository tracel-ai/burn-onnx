#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/mish/mish.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    mish_node = helper.make_node("Mish", inputs=["X"], outputs=["Y"], name="mish1")

    graph_def = helper.make_graph([mish_node], "mish_test", [X], [Y])

    model_def = helper.make_model(graph_def, producer_name="burn-onnx-test")
    model_def.opset_import[0].version = 18

    onnx.checker.check_model(model_def)

    onnx_name = "mish.onnx"
    onnx.save(model_def, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    test_input = np.array([[1.0, -1.0, 0.0, 5.0]], dtype=np.float32)
    print(f"Test input data: {test_input}")

    ref = ReferenceEvaluator(model_def)
    output = ref.run(None, {"X": test_input})[0]
    print(f"Test output data: {output}")
    print(f"Output values for Rust test: {output.flatten().tolist()}")


if __name__ == "__main__":
    main()
