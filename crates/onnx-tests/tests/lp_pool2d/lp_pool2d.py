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
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 2])

    lp_pool = helper.make_node(
        "LpPool",
        inputs=["input"],
        outputs=["output"],
        kernel_shape=[2, 3],
        strides=[1, 2],
        pads=[0, 1, 1, 0],
    )

    graph = helper.make_graph(
        nodes=[lp_pool],
        name="lp_pool2d_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = helper.make_model(
        graph,
        producer_name="burn-onnx-tests",
        opset_imports=[helper.make_opsetid("", 16)],
    )
    model.ir_version = 10
    return model


def main() -> None:
    model = create_model()
    onnx.save(model, "lp_pool2d.onnx")

    x = np.array(
        [[[[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0], [9.0, -10.0, 11.0, -12.0], [13.0, -14.0, 15.0, -16.0]]]],
        dtype=np.float32,
    )
    ref = ReferenceEvaluator(model)
    (y,) = ref.run(None, {"input": x})

    print("Saved lp_pool2d.onnx")
    print("Input:", x)
    print("Output:", y)


if __name__ == "__main__":
    main()
