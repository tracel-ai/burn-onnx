#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: deform_conv_bias.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    np.random.seed(42)

    # Shapes: X=[1,1,3,3], W=[1,1,2,2], offset=[1,8,2,2], B=[1]
    weight_data = np.random.randn(1, 1, 2, 2).astype(np.float32)
    bias_data = np.array([0.5], dtype=np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3])
    offset = helper.make_tensor_value_info("offset", TensorProto.FLOAT, [1, 8, 2, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])

    W = numpy_helper.from_array(weight_data, name="W")
    B = numpy_helper.from_array(bias_data, name="B")

    node = helper.make_node(
        "DeformConv",
        inputs=["X", "W", "offset", "B"],
        outputs=["Y"],
        kernel_shape=[2, 2],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=1,
        offset_group=1,
    )

    graph = helper.make_graph(
        [node], "deform_conv_bias_graph", [X, offset], [Y], initializer=[W, B]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    onnx.checker.check_model(model)

    file_name = "deform_conv_bias.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    # Compute expected outputs
    test_x = np.ones([1, 1, 3, 3], dtype=np.float32)
    test_offset = np.zeros([1, 8, 2, 2], dtype=np.float32)

    ref = ReferenceEvaluator(model)
    [output] = ref.run(None, {"X": test_x, "offset": test_offset})

    print("Test output shape: {}".format(output.shape))
    print("Test output: {}".format(output))
    print("Test output sum: {}".format(output.sum()))


if __name__ == "__main__":
    main()
