#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: concat_shape_with_tensor.onnx
#
# A Shape output concatenated with a runtime rank-1 tensor (a graph input, so
# it is never lifted to a constant). The tensor values have to be read back on
# host to build the fixed-size shape array (issue #438).

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 16, 8])
    extra = helper.make_tensor_value_info("extra", TensorProto.INT64, [2])
    out = helper.make_tensor_value_info("concat_out", TensorProto.INT64, [5])

    nodes = [
        helper.make_node("Shape", inputs=["x"], outputs=["shape_v"]),
        helper.make_node("Concat", inputs=["shape_v", "extra"], outputs=["concat_out"], axis=0),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="ConcatShapeWithTensor",
        inputs=[x, extra],
        outputs=[out],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model)

    onnx_name = "concat_shape_with_tensor.onnx"
    onnx.save(model, onnx_name)
    print(f"Successfully exported model to {onnx_name}")

    sess = ReferenceEvaluator(onnx_name)
    out_val, = sess.run(
        None,
        {
            "x": np.zeros((3, 16, 8), dtype=np.float32),
            "extra": np.array([9, 11], dtype=np.int64),
        },
    )
    print(f"Reference output: {out_val}")


if __name__ == "__main__":
    main()
