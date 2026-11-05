# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""Squeeze without axes preserves a dynamic non-unit batch dimension."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
graph = helper.make_graph(
    [helper.make_node("Squeeze", ["input"], ["output"])],
    "squeeze_partial",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, 3])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
onnx.checker.check_model(model)
x = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, {"input": x})[0], x.reshape(2, 3))
onnx.save(model, Path(__file__).with_suffix(".onnx"))
