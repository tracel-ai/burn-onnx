#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate custom_model.onnx: a small graph that mixes built-in ONNX operators
with two operators from a vendor domain ("example.custom") that burn-onnx does
not know.

    x [2,4] --MatMul(w)--> t0 --example.custom::ScaleBias--> t1
            --Sigmoid--> t2 --example.custom::ChannelScale(scale)--> y [2,3]

This stands in for the common real-world case: a model exported from a
framework where some layers came out as custom-domain ops. Running it through
burn-onnx requires a CustomOp hook per custom operator; the Sigmoid is also
routed to a user kernel to demonstrate OpOverride.

The expected output printed at the end is asserted by src/bin/custom_op_demo.rs.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

# Fixed weights so the example is reproducible.
np.random.seed(42)
W = np.round(np.random.randn(4, 3).astype(np.float32), 3)
CHANNEL_SCALE = np.array([1.0, 0.5, 2.0], dtype=np.float32)
SCALE, BIAS = 2.0, 0.5


def main():
    matmul = helper.make_node("MatMul", ["x", "w"], ["t0"], name="matmul1")

    # Custom op #1: attributes only. The hook reads `scale` and `bias`.
    scale_bias = helper.make_node(
        "ScaleBias",
        ["t0"],
        ["t1"],
        name="scale_bias1",
        domain="example.custom",
        scale=SCALE,
        bias=BIAS,
    )

    # Built-in op whose codegen is replaced by an OpOverride.
    sigmoid = helper.make_node("Sigmoid", ["t1"], ["t2"], name="sigmoid1")

    # Custom op #2: takes a constant initializer input. The hook reads the
    # values at codegen time via Argument::value() and inlines them.
    channel_scale = helper.make_node(
        "ChannelScale",
        ["t2", "channel_scale"],
        ["y"],
        name="channel_scale1",
        domain="example.custom",
    )

    graph = helper.make_graph(
        [matmul, scale_bias, sigmoid, channel_scale],
        "custom_op_demo",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        initializer=[
            helper.make_tensor("w", TensorProto.FLOAT, W.shape, W.flatten().tolist()),
            helper.make_tensor(
                "channel_scale",
                TensorProto.FLOAT,
                CHANNEL_SCALE.shape,
                CHANNEL_SCALE.tolist(),
            ),
        ],
    )

    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[
            helper.make_operatorsetid("", 16),
            helper.make_operatorsetid("example.custom", 1),
        ],
    )
    onnx.save(model, "custom_model.onnx")
    print("Exported custom_model.onnx")

    # Ground truth for the demo binary. There is no ONNX reference
    # implementation for custom-domain ops, so the semantics defined by the
    # hooks are mirrored here in numpy.
    x = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]], dtype=np.float32)
    t0 = x @ W
    t1 = t0 * SCALE + BIAS
    t2 = 1.0 / (1.0 + np.exp(-t1))
    y = t2 * CHANNEL_SCALE
    np.set_printoptions(precision=6, suppress=False)
    print(f"input:    {x.tolist()}")
    print(f"expected: {y.tolist()}")


if __name__ == "__main__":
    main()
