#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_shape_runtime_bounds_concat.onnx
#
# A runtime-bound Shape slice (rank-1 i64 tensor, length known only at
# inference time) concatenated with a constant-bound Shape slice (fixed-size
# array). The two representations have to be unified, otherwise the generated
# code indexes a tensor as if it were an array (issue #438).

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    key = helper.make_tensor_value_info("key", TensorProto.FLOAT, [None, None, 64])
    end_in = helper.make_tensor_value_info("end_in", TensorProto.INT64, [1])
    out = helper.make_tensor_value_info("concat_shape", TensorProto.INT64, [None])

    starts_head = helper.make_tensor("starts_head", TensorProto.INT64, [1], [0])
    ends_head = helper.make_tensor("ends_head", TensorProto.INT64, [1], [1])
    starts_tail = helper.make_tensor("starts_tail", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Shape", inputs=["key"], outputs=["shape_v"]),
        # Constant bounds: stays a fixed-size shape array.
        helper.make_node(
            "Slice",
            inputs=["shape_v", "starts_head", "ends_head"],
            outputs=["head"],
        ),
        # Runtime end bound: length is only known at inference time.
        helper.make_node(
            "Slice",
            inputs=["shape_v", "starts_tail", "end_in"],
            outputs=["tail"],
        ),
        helper.make_node("Concat", inputs=["head", "tail"], outputs=["concat_shape"], axis=0),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="SliceShapeRuntimeBoundsConcat",
        inputs=[key, end_in],
        outputs=[out],
        initializer=[starts_head, ends_head, starts_tail],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model)

    onnx_name = "slice_shape_runtime_bounds_concat.onnx"
    onnx.save(model, onnx_name)
    print(f"Successfully exported model to {onnx_name}")

    # key shape (4, 7, 64), tail slice [1:3] => [4] ++ [7, 64]
    sess = ReferenceEvaluator(onnx_name)
    out_val, = sess.run(
        None,
        {
            "key": np.zeros((4, 7, 64), dtype=np.float32),
            "end_in": np.array([3], dtype=np.int64),
        },
    )
    print(f"Reference output: {out_val}")


if __name__ == "__main__":
    main()
