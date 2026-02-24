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
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y_scale = helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, [1])
    y_zero = helper.make_tensor_value_info("y_zero", TensorProto.INT16, [1])
    y = helper.make_tensor_value_info("y", TensorProto.INT16, [1, 4])

    q = helper.make_node(
        "QuantizeLinear",
        inputs=["x", "y_scale", "y_zero"],
        outputs=["y"],
    )

    graph = helper.make_graph(
        [q],
        "quantize_linear_graph",
        [x, y_scale, y_zero],
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
    onnx.save(model, "quantize_linear.onnx")

    x = np.array([[-1.0, 0.0, 1.0, 2.1]], dtype=np.float32)
    y_scale = np.array([0.5], dtype=np.float32)
    y_zero = np.array([3], dtype=np.int16)
    ref = ReferenceEvaluator(model)
    (y,) = ref.run(None, {"x": x, "y_scale": y_scale, "y_zero": y_zero})

    print("Saved quantize_linear.onnx")
    print("Input:", x)
    print("Output:", y)


if __name__ == "__main__":
    main()
