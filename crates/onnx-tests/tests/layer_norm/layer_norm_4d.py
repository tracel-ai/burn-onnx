#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate LayerNormalization model with 4D input [2, 2, 3, 4], axis=-1."""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 17


def main():
    num_features = 4

    # Scale and bias initializers
    scale_init = numpy_helper.from_array(
        np.ones(num_features, dtype=np.float32), name="scale"
    )
    bias_init = numpy_helper.from_array(
        np.zeros(num_features, dtype=np.float32), name="bias"
    )

    ln_node = helper.make_node(
        "LayerNormalization",
        ["X", "scale", "bias"],
        ["Y"],
        axis=-1,
        epsilon=1e-5,
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2, 3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 3, 4])

    graph = helper.make_graph(
        [ln_node],
        "layer_norm_4d",
        [X],
        [Y],
        initializer=[scale_init, bias_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "layer_norm_4d.onnx")

    # Compute expected outputs
    test_input = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
    ref = ReferenceEvaluator(model)
    (output_val,) = ref.run(None, {"X": test_input})
    print("Input:", repr(test_input))
    print("\nExpected output:")
    print(repr(output_val))


if __name__ == "__main__":
    main()
