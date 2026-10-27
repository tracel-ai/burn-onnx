#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: maxpool3d.onnx
#
# MaxPool 3D on a [1, 2, 4, 5, 5] input:
# - y: kernel 2x3x3, strides 2x2x2, asymmetric pads (front 1, top 0, left 1, back 0,
#   bottom 1, right 1), ceil_mode, dilation 1x2x1;
# - y_indices_row / y_indices_col: same pooling with the Indices output, storage_order 0
#   and 1 (positions across the whole flattened input);
# - y_same: SAME_UPPER on the same input, kernel 3x2x3, strides 2x2x2.
# Expected values from onnx.reference.ReferenceEvaluator, checked against ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

POOL = dict(kernel_shape=[2, 3, 3], strides=[2, 2, 2], pads=[1, 0, 1, 0, 1, 1], ceil_mode=1, dilations=[1, 2, 1])


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 5, 5])
    nodes = [
        helper.make_node("MaxPool", ["x"], ["y"], **POOL),
        helper.make_node("MaxPool", ["x"], ["y_row", "y_indices_row"], storage_order=0, **POOL),
        helper.make_node("MaxPool", ["x"], ["y_col", "y_indices_col"], storage_order=1, **POOL),
        helper.make_node("MaxPool", ["x"], ["y_same"], kernel_shape=[3, 2, 3], strides=[2, 2, 2], auto_pad="SAME_UPPER"),
    ]
    names = ["y", "y_row", "y_indices_row", "y_col", "y_indices_col", "y_same"]
    types = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT]
    outputs = [helper.make_tensor_value_info(n, t, None) for n, t in zip(names, types)]
    for out in outputs:
        out.type.tensor_type.shape.dim.add()  # rank only; shapes come from inference
    graph = helper.make_graph(nodes, "maxpool3d_graph", [x], outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    for out in model.graph.output:
        out.type.tensor_type.ClearField("shape")
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, "maxpool3d.onnx")

    rng = np.random.default_rng(42)
    data = rng.permutation(2 * 4 * 5 * 5).astype(np.float32).reshape(1, 2, 4, 5, 5)
    reference = ReferenceEvaluator(model).run(None, {"x": data})
    runtime = ort.InferenceSession(model.SerializeToString()).run(None, {"x": data})
    for name, ref, rt in zip(names, reference, runtime):
        assert ref.shape == rt.shape and (ref == rt).all(), name
        print(name, ref.shape, ref.flatten().tolist())


if __name__ == "__main__":
    main()
