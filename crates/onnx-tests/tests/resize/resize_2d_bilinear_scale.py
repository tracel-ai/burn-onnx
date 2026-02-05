#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: resize_2d_bilinear_scale.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 17


def main():
    node0 = helper.make_node(
        "Constant",
        [],
        ["/Constant_output_0"],
        value=numpy_helper.from_array(
            np.array([1.0, 1.0, 1.5, 1.5], dtype=np.float32).reshape([4]), name="value"
        ),
    )
    node1 = helper.make_node(
        "Resize",
        ["input", "", "/Constant_output_0"],
        ["output"],
        coordinate_transformation_mode="align_corners",
        cubic_coeff_a=-0.75,
        mode="linear",
        nearest_mode="floor",
    )

    inp_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, 1, 6, 6]
    )

    out_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1, 9, 9]
    )

    graph = helper.make_graph(
        [node0, node1],
        "main_graph",
        [inp_input],
        [out_output],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "resize_2d_bilinear_scale.onnx")
    print(f"Finished exporting model to resize_2d_bilinear_scale.onnx")


if __name__ == "__main__":
    main()
