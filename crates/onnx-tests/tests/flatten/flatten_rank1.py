#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

# used to generate model: flatten_rank1.onnx

# Flatten rank-1 input [5] with axis=0 -> output [1, 5]
# Validates that Flatten accepts rank-1 tensors per the ONNX specification.

import onnx
import onnx.helper


def build_model():
    return onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(
            name="FlattenRank1Graph",
            nodes=[
                onnx.helper.make_node(
                    "Flatten",
                    inputs=["a"],
                    outputs=["b"],
                    axis=0,
                ),
            ],
            inputs=[
                onnx.helper.make_value_info(
                    name="a",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[5]
                    ),
                ),
            ],
            outputs=[
                onnx.helper.make_value_info(
                    name="b",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[1, 5]
                    ),
                )
            ],
        ),
    )


if __name__ == "__main__":
    onnx_model = build_model()
    file_name = "flatten_rank1.onnx"

    # Ensure valid ONNX:
    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")
