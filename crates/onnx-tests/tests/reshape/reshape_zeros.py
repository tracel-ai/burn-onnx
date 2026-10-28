#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///

"""Exercise zero semantics with constant, tensor, and host-shape targets."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def generate(name, allowzero, data, target, shape_source):
    attributes = {} if allowzero is None else {"allowzero": allowzero}
    nodes = [
        helper.make_node("Shape", ["shape_source"], ["host_shape"]),
        helper.make_node(
            "Reshape", ["data", "constant_shape"], ["static_out"], **attributes
        ),
        helper.make_node("Reshape", ["data", "target"], ["tensor_out"], **attributes),
        helper.make_node("Reshape", ["data", "host_shape"], ["shape_out"], **attributes),
    ]
    graph = helper.make_graph(
        nodes,
        name,
        [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [None] * data.ndim),
            helper.make_tensor_value_info("target", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("shape_source", TensorProto.FLOAT, [None, None]),
        ],
        [
            helper.make_tensor_value_info(out, TensorProto.FLOAT, [None, None])
            for out in ("static_out", "tensor_out", "shape_out")
        ],
        [numpy_helper.from_array(target, name="constant_shape")],
    )
    model = helper.make_model(
        graph, ir_version=8, opset_imports=[helper.make_opsetid("", 16)]
    )
    onnx.checker.check_model(model)
    outputs = ReferenceEvaluator(model).run(
        None, {"data": data, "target": target, "shape_source": shape_source}
    )
    expected_shape = (3, 0) if allowzero else (2, 3)
    expected = data.reshape(expected_shape)
    for output in outputs:
        np.testing.assert_array_equal(output, expected, strict=True)
    onnx.save(model, Path(__file__).with_name(f"{name}.onnx"))
    print(f"{name}: all three outputs match {expected_shape}: {expected.tolist()}")


if __name__ == "__main__":
    generate(
        "reshape_zero_placeholders",
        None,  # Omitted allowzero defaults to dimension copying.
        np.array([[[1, -2, 4]], [[7, 0.5, -3]]], dtype=np.float32),
        np.array([0, -1], dtype=np.int64),
        np.empty((0, 3), dtype=np.float32),
    )
    generate(
        "reshape_literal_zeros",
        1,
        np.empty((0, 3), dtype=np.float32),
        np.array([3, 0], dtype=np.int64),
        np.empty((3, 0), dtype=np.float32),
    )
