#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: avgpool3d.onnx
#
# AveragePool 3D on an input whose dims are only known at run time ([N, C, D, H, W]):
# - y_include / y_exclude: kernel 2x3x2, strides 2x2x2, asymmetric pads (front 1, top 0,
#   left 1, back 0, bottom 1, right 0), ceil_mode, count_include_pad 1 and 0;
# - y_same: SAME_LOWER, kernel 3x2x3, strides 2x2x2, so the pads are computed at run time
#   from the input dims.
# Run on a [1, 2, 4, 5, 5] input. Expected values from onnx.reference.ReferenceEvaluator,
# checked against ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

POOL = dict(kernel_shape=[2, 3, 2], strides=[2, 2, 2], pads=[1, 0, 1, 0, 1, 0], ceil_mode=1)


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", "C", "D", "H", "W"])
    nodes = [
        helper.make_node("AveragePool", ["x"], ["y_include"], count_include_pad=1, **POOL),
        helper.make_node("AveragePool", ["x"], ["y_exclude"], count_include_pad=0, **POOL),
        helper.make_node("AveragePool", ["x"], ["y_same"], kernel_shape=[3, 2, 3], strides=[2, 2, 2], auto_pad="SAME_LOWER"),
    ]
    names = ["y_include", "y_exclude", "y_same"]
    outputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, ["N", "C", "OD", "OH", "OW"]) for n in names]
    graph = helper.make_graph(nodes, "avgpool3d_graph", [x], outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, "avgpool3d.onnx")

    rng = np.random.default_rng(42)
    data = rng.standard_normal((1, 2, 4, 5, 5)).astype(np.float32)
    reference = ReferenceEvaluator(model).run(None, {"x": data})
    runtime = ort.InferenceSession(model.SerializeToString()).run(None, {"x": data})
    for name, ref, rt in zip(names, reference, runtime):
        assert ref.shape == rt.shape, name
        assert np.allclose(ref, rt, atol=1e-6), (name, float(np.abs(ref - rt).max()))
        print(name, ref.shape, np.round(ref, 6).flatten().tolist())


if __name__ == "__main__":
    main()
