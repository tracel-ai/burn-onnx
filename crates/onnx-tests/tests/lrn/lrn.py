#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "numpy==2.2.4",
#   "onnx==1.19.0",
# ]
# ///

# Used to generate model: `lrn_default_size3.onnx`, `lrn_custom_size3.onnx`, and `lrn_custom_size2.onnx`
# and expected values for the integration test suite in `./mod.rs`.
#
# NOTE: Do not regenerate the expected values using `onnx`'s stock `ReferenceEvaluator`
#       until https://github.com/onnx/onnx/issues/7805 has been resolved.
#       Expected values in the integration test suite were generated
#       using a manually patched LRN op.
# TODO: Once resolved, remove these "NOTE" and "TODO" comments.

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model(size, alpha=0.0001, beta=0.75, bias=1.0, suffix=None):
    np.random.seed(42)

    # Shape: [N, C, H, W] — LRN normalizes over the channel dimension
    input_shape = [1, 5, 3, 3]
    test_input = np.random.randn(*input_shape).astype(np.float32)

    # Build LRN node directly (avoid PyTorch export which decomposes into primitives)
    node = helper.make_node(
        "LRN",
        inputs=["input"],
        outputs=["output"],
        size=size,
        alpha=alpha,
        beta=beta,
        bias=bias,
    )

    input_value_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    output_value_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, input_shape
    )

    graph = helper.make_graph(
        [node], "lrn_graph", [input_value_info], [output_value_info]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model)

    file_name = f"lrn_{suffix}.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    # Use ReferenceEvaluator as ground truth
    session = ReferenceEvaluator(model)
    output = session.run(None, {"input": test_input})[0]

    print("Test input shape: {}".format(test_input.shape))
    print("Test input:\n{}".format(test_input))
    print("Test output shape: {}".format(output.shape))
    print("Test output:\n{}".format(output))


if __name__ == "__main__":
    build_model(size=3, suffix="default_size3")
    # Non-default params: larger alpha and different beta make normalization effect clearly visible
    build_model(size=3, alpha=0.1, beta=0.5, bias=0.5, suffix="custom_size3")
    build_model(size=2, alpha=0.1, beta=0.5, bias=0.5, suffix="custom_size2")
