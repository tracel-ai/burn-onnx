#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: deform_conv_runtime_bias.onnx
#
# Constant weight (initializer) with a runtime bias (graph input). Weight and
# bias must be lifted together, so this takes the functional path instead of
# the module path, which has no way to take the runtime bias.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    np.random.seed(42)

    # Shapes: X=[1,1,3,3], W=[1,1,2,2], offset=[1,8,2,2], B=[1]
    W = numpy_helper.from_array(
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32), name="W"
    )
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
        [node],
        "deform_conv_runtime_bias_graph",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3]),
            helper.make_tensor_value_info("offset", TensorProto.FLOAT, [1, 8, 2, 2]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])],
        initializer=[W],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    onnx.checker.check_model(model)

    file_name = "deform_conv_runtime_bias.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")

    test_x = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    test_offset = np.zeros([1, 8, 2, 2], dtype=np.float32)
    test_bias = np.array([0.5], dtype=np.float32)
    (output,) = ReferenceEvaluator(model).run(
        None, {"X": test_x, "offset": test_offset, "B": test_bias}
    )
    print(f"Test output: {repr(output)}")


if __name__ == "__main__":
    main()
