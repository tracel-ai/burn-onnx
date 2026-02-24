#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def create_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [1, 4])
    x_scale = helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])

    dq = helper.make_node(
        "DequantizeLinear",
        inputs=["x", "x_scale"],
        outputs=["y"],
    )

    graph = helper.make_graph(
        [dq],
        "dequantize_linear_graph",
        [x, x_scale],
        [y],
    )

    model = helper.make_model(
        graph,
        producer_name="burn-onnx-tests",
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10
    return model


def main() -> None:
    model = create_model()
    onnx.save(model, "dequantize_linear.onnx")

    x = np.array([[2, 4, 6, 10]], dtype=np.int32)
    x_scale = np.array([0.5], dtype=np.float32)
    ref = ReferenceEvaluator(model)
    (y,) = ref.run(None, {"x": x, "x_scale": x_scale})

    print("Saved dequantize_linear.onnx")
    print("Input:", x)
    print("Output:", y)


if __name__ == "__main__":
    main()
