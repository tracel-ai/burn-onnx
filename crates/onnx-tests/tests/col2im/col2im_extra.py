#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate models: col2im_asym.onnx, col2im_1d.onnx
#
# Col2Im with asymmetric pads plus stride and dilation, and a 1D Col2Im.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def build(name, input_shape, output_shape, image_shape, block_shape, **attrs):
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    node = helper.make_node(
        "Col2Im", ["input", "image_shape", "block_shape"], ["output"], **attrs
    )
    graph = helper.make_graph(
        [node],
        name,
        [x],
        [y],
        initializer=[
            numpy_helper.from_array(np.array(image_shape, dtype=np.int64), "image_shape"),
            numpy_helper.from_array(np.array(block_shape, dtype=np.int64), "block_shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, f"{name}.onnx")

    data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    [out] = ReferenceEvaluator(model).run(None, {"input": data})
    print(f"{name}: {out.tolist()}")


def main():
    # image 4x5, kernel 2x2, stride 2x1, dilation 1x2, pads top=1 left=0 bottom=0 right=2.
    # Padded 5x7; blocks along H: (5 - 2) // 2 + 1 = 2, along W: (7 - 3) // 1 + 1 = 5.
    build(
        "col2im_asym",
        [1, 4, 10],
        [1, 1, 4, 5],
        [4, 5],
        [2, 2],
        strides=[2, 1],
        dilations=[1, 2],
        pads=[1, 0, 0, 2],
    )
    # 1D: image 6, kernel 3, stride 1 -> 4 blocks, 2 channels.
    build("col2im_1d", [1, 6, 4], [1, 2, 6], [6], [3])


if __name__ == "__main__":
    main()
