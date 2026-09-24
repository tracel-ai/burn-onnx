#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: col2im_runtime.onnx
#
# Col2Im with image_shape read at run time: a 2D fold with asymmetric pads
# [top=1, left=0, bottom=0, right=2] and a 1D fold.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    def const(name, value):
        return helper.make_node(
            "Constant", [], [name],
            value=numpy_helper.from_array(np.array(value, dtype=np.int64), name=name),
        )

    nodes = [
        const("block2d", [2, 2]),
        const("block1d", [3]),
        helper.make_node(
            "Col2Im", ["x2d", "image2d", "block2d"], ["y2d"], pads=[1, 0, 0, 2]
        ),
        helper.make_node("Col2Im", ["x1d", "image1d", "block1d"], ["y1d"]),
    ]
    graph = helper.make_graph(
        nodes,
        "col2im_runtime",
        [
            helper.make_tensor_value_info("x2d", TensorProto.FLOAT, [1, 4, 12]),
            helper.make_tensor_value_info("image2d", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("x1d", TensorProto.FLOAT, [1, 3, 3]),
            helper.make_tensor_value_info("image1d", TensorProto.INT64, [1]),
        ],
        [
            helper.make_tensor_value_info("y2d", TensorProto.FLOAT, [1, 1, 3, 3]),
            helper.make_tensor_value_info("y1d", TensorProto.FLOAT, [1, 1, 5]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, "col2im_runtime.onnx")

    inputs = {
        "x2d": np.arange(48, dtype=np.float32).reshape(1, 4, 12),
        "image2d": np.array([3, 3], dtype=np.int64),
        "x1d": np.arange(9, dtype=np.float32).reshape(1, 3, 3),
        "image1d": np.array([5], dtype=np.int64),
    }
    y2d, y1d = ReferenceEvaluator(model).run(None, inputs)
    print(f"y2d: {y2d.tolist()}")
    print(f"y1d: {y1d.tolist()}")


if __name__ == "__main__":
    main()
