#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_shape_runtime_bounds_concat_reshape.onnx
#
# The pattern reported in issue #438: head-splitting shape arithmetic where a
# runtime-bound Shape slice is concatenated with a constant-bound one and the
# result drives a Reshape. Exercises the whole chain, not just the Concat, since
# the Concat output type is what Reshape reads to pick its output rank.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, None, 64])
    end_in = helper.make_tensor_value_info("end_in", TensorProto.INT64, [1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, None, None])

    initializers = [
        helper.make_tensor("starts_head", TensorProto.INT64, [1], [0]),
        helper.make_tensor("ends_head", TensorProto.INT64, [1], [1]),
        helper.make_tensor("starts_tail", TensorProto.INT64, [1], [1]),
    ]

    nodes = [
        helper.make_node("Shape", inputs=["x"], outputs=["shape_v"]),
        helper.make_node("Slice", inputs=["shape_v", "starts_head", "ends_head"], outputs=["head"]),
        helper.make_node("Slice", inputs=["shape_v", "starts_tail", "end_in"], outputs=["tail"]),
        helper.make_node("Concat", inputs=["head", "tail"], outputs=["new_shape"], axis=0),
        helper.make_node("Reshape", inputs=["x", "new_shape"], outputs=["out"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="SliceShapeRuntimeBoundsConcatReshape",
        inputs=[x, end_in],
        outputs=[out],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model)

    onnx_name = "slice_shape_runtime_bounds_concat_reshape.onnx"
    onnx.save(model, onnx_name)
    print(f"Successfully exported model to {onnx_name}")

    # x is (4, 7, 64), tail slice [1:3] => reshape to [4] ++ [7, 64], a no-op
    sess = ReferenceEvaluator(onnx_name)
    out_val, = sess.run(
        None,
        {
            "x": np.zeros((4, 7, 64), dtype=np.float32),
            "end_in": np.array([3], dtype=np.int64),
        },
    )
    print(f"Reference output shape: {out_val.shape}")


if __name__ == "__main__":
    main()
