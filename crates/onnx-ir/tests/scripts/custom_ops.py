#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate an ONNX test model containing custom (non-built-in) operators.

The model exercises the parser's domain-aware fallback:
- two identical nodes from a custom domain (CSE must NOT merge them)
- a custom-domain node with scalar/list/string attributes and a constant input
- an unknown op_type in the default ONNX domain

Graph (all tensors float32 [2, 4]):

    input -> Relu -> r_out
    FftLike(r_out, window) -> f1        # my.custom.domain, opset 3
    FftLike(r_out, window) -> f2        # identical twin (CSE bait)
    Add(f1, f2) -> a_out
    MyUnknownOp(a_out) -> output        # default domain, unknown op_type
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import os

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def create_custom_ops_model():
    window = numpy_helper.from_array(
        np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32), name="window"
    )

    relu = helper.make_node("Relu", ["input"], ["r_out"], name="relu1")

    def fft_like(name, output):
        return helper.make_node(
            "FftLike",
            ["r_out", "window"],
            [output],
            name=name,
            domain="my.custom.domain",
            n_fft=1024,
            scale=0.5,
            mode="real",
            axes=[0, 1],
        )

    fft1 = fft_like("fft1", "f1")
    fft2 = fft_like("fft2", "f2")
    add = helper.make_node("Add", ["f1", "f2"], ["a_out"], name="add1")
    unknown = helper.make_node("MyUnknownOp", ["a_out"], ["output"], name="unknown1")

    graph = helper.make_graph(
        [relu, fft1, fft2, add, unknown],
        "custom_ops_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
        initializer=[window],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 16),
            helper.make_opsetid("my.custom.domain", 3),
        ],
    )
    model.ir_version = 8

    output_path = os.path.join(FIXTURES_DIR, "custom_ops.onnx")
    onnx.save(model, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    create_custom_ops_model()
