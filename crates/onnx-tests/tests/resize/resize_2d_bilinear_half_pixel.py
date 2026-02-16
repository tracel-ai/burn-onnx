#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: resize_2d_bilinear_half_pixel.onnx
# Tests bilinear interpolation with half_pixel coordinate transformation mode.

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
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32).reshape([4]), name="value"
        ),
    )
    node1 = helper.make_node(
        "Resize",
        ["input", "", "/Constant_output_0"],
        ["output"],
        coordinate_transformation_mode="half_pixel",
        mode="linear",
    )

    inp_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, 1, 4, 4]
    )

    out_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1, 8, 8]
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

    onnx.save(model, "resize_2d_bilinear_half_pixel.onnx")

    # Verify with onnx.reference.ReferenceEvaluator
    from onnx.reference import ReferenceEvaluator

    test_input = np.array(
        [[[[0.0, 1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0, 7.0],
           [8.0, 9.0, 10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]]],
        dtype=np.float32,
    )

    sess = ReferenceEvaluator(model)
    result = sess.run(None, {"input": test_input})

    print(f"Output shape: {result[0].shape}")
    print(f"Output:\n{result[0]}")
    print(f"Output sum: {result[0].sum()}")


if __name__ == "__main__":
    main()
