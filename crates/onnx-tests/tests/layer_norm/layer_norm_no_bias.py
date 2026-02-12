#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate LayerNormalization model without bias (2 inputs only)."""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 17


def main():
    num_features = 4

    # Scale (gamma) initializer - ones
    scale_init = numpy_helper.from_array(
        np.ones(num_features, dtype=np.float32), name="scale"
    )

    # LayerNormalization node with only 2 inputs (no bias)
    ln_node = helper.make_node(
        "LayerNormalization",
        ["X", "scale"],
        ["Y"],
        axis=-1,
        epsilon=1e-5,
    )

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])

    graph = helper.make_graph(
        [ln_node],
        "layer_norm_no_bias",
        [X],
        [Y],
        initializer=[scale_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "layer_norm_no_bias.onnx")

    # Compute expected outputs
    test_input = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    ref = ReferenceEvaluator(model)
    (output_val,) = ref.run(None, {"X": test_input})
    print("Input:", repr(test_input))
    print("\nExpected output:")
    print(repr(output_val))


if __name__ == "__main__":
    main()
