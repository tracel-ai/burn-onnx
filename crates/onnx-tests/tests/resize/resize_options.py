#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""Exercise fixed nearest presets, fractional scales and a custom cubic coefficient."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 4])
scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])
scale_values = np.array([1, 1, 1, 0.6], dtype=np.float32)
initializers = [
    numpy_helper.from_array(scale_values, name="fixed_scales"),
    numpy_helper.from_array(np.array([1, 1, 1, 2], dtype=np.int64), name="sizes"),
]
nearest = dict(mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
cubic = dict(mode="cubic", coordinate_transformation_mode="half_pixel", cubic_coeff_a=-0.5)
nodes = [
    helper.make_node("Resize", ["x", "", "fixed_scales"], ["static_nearest"], **nearest),
    helper.make_node("Resize", ["x", "", "scales"], ["runtime_nearest"], **nearest),
    helper.make_node("Resize", ["x", "", "", "sizes"], ["exact"], mode="nearest",
                     coordinate_transformation_mode="half_pixel", nearest_mode="round_prefer_ceil"),
    helper.make_node("Resize", ["x", "", "fixed_scales"], ["static_cubic"], **cubic),
    helper.make_node("Resize", ["x", "", "scales"], ["runtime_cubic"], **cubic),
]
outputs = [helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, [1, 1, 1, 2]) for node in nodes]
model = helper.make_model(helper.make_graph(nodes, "resize_options", [x, scales], outputs, initializers),
                          opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_suffix(".onnx"))
values = ReferenceEvaluator(model).run(None, {
    "x": np.array([10, 20, 30, 40], dtype=np.float32).reshape(1, 1, 1, 4),
    "scales": scale_values,
})
for output, value in zip(outputs, values):
    print(output.name, value.tolist())
