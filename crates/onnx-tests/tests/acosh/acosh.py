#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

# used to generate model: acosh.onnx

import onnx
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 22)],
        graph=onnx.helper.make_graph(
            name="main_graph",
            nodes=[
                onnx.helper.make_node(
                    "Acosh", inputs=["input1"], outputs=["output1"], name="/Acosh"
                ),
            ],
            inputs=[
                onnx.helper.make_value_info(
                    name="input1",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[1, 1, 1, 4]
                    ),
                ),
            ],
            outputs=[
                onnx.helper.make_value_info(
                    name="output1",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[1, 1, 1, 4]
                    ),
                )
            ],
        ),
    )


def main():
    onnx_model = build_model()
    file_name = "acosh.onnx"

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # acosh domain is [1, inf)
    test_input = np.array([[[[1.0, 2.0, 5.0, 10.0]]]], dtype=np.float32)
    ref = ReferenceEvaluator(onnx_model)
    result = ref.run(None, {"input1": test_input})
    print(f"Test input data: {test_input}")
    print(f"Test output data: {result[0]}")


if __name__ == "__main__":
    main()
