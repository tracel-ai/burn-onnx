#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# LpPool opset 1 declares `p` as a FLOAT attribute (INT from opset 2 on), so `p` may
# be fractional; this model uses 1.5.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def create_model() -> onnx.ModelProto:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 6])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3])

    lp_pool = helper.make_node(
        "LpPool",
        inputs=["input"],
        outputs=["output"],
        kernel_shape=[2],
        strides=[2],
        p=1.5,
    )

    graph = helper.make_graph(
        nodes=[lp_pool],
        name="lp_pool1d_opset1_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = helper.make_model(
        graph,
        producer_name="burn-onnx-tests",
        opset_imports=[helper.make_opsetid("", 1)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    np.random.seed(42)
    model = create_model()
    onnx.save(model, "lp_pool1d_opset1.onnx")

    x = np.random.randn(1, 1, 6).astype(np.float32)
    ref = ReferenceEvaluator(model)
    (y,) = ref.run(None, {"input": x})

    # Cross-check against the spec formula so a reference evaluator that truncated
    # p to an integer would be caught.
    windows = np.abs(x.astype(np.float64)).reshape(1, 1, 3, 2)
    expected = ((windows**1.5).sum(axis=3) ** (1.0 / 1.5)).astype(np.float32)
    np.testing.assert_allclose(y, expected, rtol=1e-5)

    print("Saved lp_pool1d_opset1.onnx")
    print("Input:", x)
    print("Output:", y)


if __name__ == "__main__":
    main()
